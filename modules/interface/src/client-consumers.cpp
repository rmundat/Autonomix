#include "common/ad-utils/data-utils.h"
#include "interface/src/client-consumers.h"
#include "planning/mission-planner/src/navi-smoothing.h"
#include "control/src/setup-mpc.h"
#include "control/src/motion-control.h"

using namespace std;
namespace py = pybind11;

CurrentEgoControl currentEgoControl {
    .steer = 0.0,
    .throttle = 0.0,
    .brake = 0.0,
    .x = 0.0,
    .y = 0.0,
    .psi = 0.0,
    .v = 0.0,
    .cte = 0.0,
    .epsi = 0.0,
    .checkpoint = false,
};

CurrentEgoState currentEgoState;
EgoParameters egoParams;
WaypointTransform wpTransform;
PlotData plotData;
vector<vector<double>> waypoints;
mutex currentEgoControlMutex;
set<pair<double, double>> wpUnique;

carla::geom::Location convertPycarlaLocationToCpp(py::object pyLocation) {
    currentEgoState.x = pyLocation.attr("x").cast<double>();
    currentEgoState.y = pyLocation.attr("y").cast<double>();
    currentEgoState.z = pyLocation.attr("z").cast<double>();
    return carla::geom::Location(currentEgoState.x, currentEgoState.y, currentEgoState.z);
}

carla::geom::Rotation convertPycarlaRotationToCpp(py::object pyOrientation) {
    currentEgoState.pitch = pyOrientation.attr("pitch").cast<double>();
    currentEgoState.yaw = pyOrientation.attr("yaw").cast<double>();
    currentEgoState.roll = pyOrientation.attr("roll").cast<double>();
    return carla::geom::Rotation(currentEgoState.pitch, currentEgoState.yaw, currentEgoState.roll);
}

carla::geom::Vector3D convertPycarlaRotationRateToCpp(py::object pyRotationRate) {
    currentEgoState.rotationX = pyRotationRate.attr("x").cast<double>();
    currentEgoState.rotationY = pyRotationRate.attr("y").cast<double>();
    currentEgoState.rotationZ = pyRotationRate.attr("z").cast<double>();
    return carla::geom::Vector3D(currentEgoState.rotationX, currentEgoState.rotationY, currentEgoState.rotationZ);
}

carla::geom::Vector3D convertPycarlaVector3DToCpp(py::object pyVelocity) {
    double x = pyVelocity.attr("x").cast<double>();
    double y = pyVelocity.attr("y").cast<double>();
    double z = pyVelocity.attr("z").cast<double>();
    return carla::geom::Vector3D(x, y, z);
}

EgoAttributes getVehiclePhysicsAttributes(py::object pyVehicleAttributes) {
    EgoAttributes egoAttributes;
    
    egoAttributes.mass = pyVehicleAttributes.attr("mass").cast<double>();
    egoAttributes.moi = pyVehicleAttributes.attr("moi").cast<double>();
    egoAttributes.cd = pyVehicleAttributes.attr("drag_coefficient").cast<double>();
    
    py::object pyCom = pyVehicleAttributes.attr("center_of_mass");
    egoAttributes.centerOfMass.x = pyCom.attr("x").cast<double>();
    egoAttributes.centerOfMass.y = pyCom.attr("y").cast<double>();
    egoAttributes.centerOfMass.z = pyCom.attr("z").cast<double>();
    
    py::list pyWheels = pyVehicleAttributes.attr("wheels").cast<py::list>();
    vector<vector<double>*> wheelPhysicsPtrs = {
        &egoAttributes.wheelPhysics.frontLeft,
        &egoAttributes.wheelPhysics.frontRight,
        &egoAttributes.wheelPhysics.rearLeft,
        &egoAttributes.wheelPhysics.rearRight
    };

    int wheelIndex = 0;
    for (const auto& wheel : pyWheels) {
        double tireFriction = wheel.attr("tire_friction").cast<double>();
        double dampingRate = wheel.attr("damping_rate").cast<double>();
        double maxSteerAngle = wheel.attr("max_steer_angle").cast<double>();
        double radius = wheel.attr("radius").cast<double>();
        double longStiffValue = wheel.attr("long_stiff_value").cast<double>();
        double latStiffMaxLoad = wheel.attr("lat_stiff_max_load").cast<double>();
        double latStiffValue = wheel.attr("lat_stiff_value").cast<double>();
        py::object wheelPos = wheel.attr("position");  //Wheel position in cm
        double wheelPositionX = wheelPos.attr("x").cast<double>();
        double wheelPositionY = wheelPos.attr("y").cast<double>();
        double wheelPositionZ = wheelPos.attr("z").cast<double>();

        std::vector<double> wheelAttr = {
            tireFriction, dampingRate, maxSteerAngle, radius,
            longStiffValue, latStiffMaxLoad, latStiffValue,
            wheelPositionX, wheelPositionY, wheelPositionZ
        };

        if (wheelIndex < wheelPhysicsPtrs.size()) {
            *wheelPhysicsPtrs[wheelIndex] = wheelAttr;
        }
        wheelIndex++;
    }

    return egoAttributes;
}

vector<double> getVehicleControlValues(py::object pyVehicleControl) {
    currentEgoState.steer = pyVehicleControl.attr("steer").cast<double>();
    currentEgoState.throttle = pyVehicleControl.attr("throttle").cast<double>();
    currentEgoState.brake = pyVehicleControl.attr("brake").cast<double>();
    vector<double> controlValues = {currentEgoState.steer, currentEgoState.throttle, currentEgoState.brake};
    return controlValues;
}

vector<double> convertPycarlaWaypointsToCpp(const py::handle& pyWaypoint) {
    double wpx = pyWaypoint.attr("x").cast<double>();
    double wpy = pyWaypoint.attr("y").cast<double>();
    double wpz = pyWaypoint.attr("z").cast<double>();

    if (wpUnique.insert({wpx, wpy}).second) {
        wpTransform.x.push_back(wpx);
        wpTransform.y.push_back(wpy);
        wpTransform.z.push_back(wpz);
    }

    vector<double> waypointLocation = {wpx, wpy, wpz};
    return waypointLocation;
}

void processVehicleInformation(py::dict egoVehicleInformation, py::list waypointsList) {

    // Extract information from the dictionary and process it
    auto position = convertPycarlaLocationToCpp(egoVehicleInformation["position"]);
    auto orientation = convertPycarlaRotationToCpp(egoVehicleInformation["orientation"]);
    auto angular = convertPycarlaRotationRateToCpp(egoVehicleInformation["angular"]);
    auto velocity = convertPycarlaVector3DToCpp(egoVehicleInformation["velocity"]);
    auto attributes = getVehiclePhysicsAttributes(egoVehicleInformation["attributes"]);
    auto control = getVehicleControlValues(egoVehicleInformation["control"]);

    // Process the list of waypoints
    size_t index = 0;
    size_t waypointsListSize = waypointsList.size();
    while (waypoints.size() < waypointsListSize) {
        try {
            waypoints.push_back(convertPycarlaWaypointsToCpp(waypointsList[index]));
        } catch (const std::exception& e) {
            cerr << "Exception occurred while converting waypoints: " << e.what() << endl;
            break;
        }
        ++index;
    }

    // Instantiate vehicle object and update its state
    egoParams.m = attributes.mass;
    currentEgoState.psi = deg2rad(orientation.yaw);
    currentEgoState.r = currentEgoState.rotationZ;

    // Ego direction in the horizontal plane
    currentEgoState.forwardVectorGlobal.x = cos(currentEgoState.psi);
    currentEgoState.forwardVectorGlobal.y = sin(currentEgoState.psi);
    currentEgoState.forwardVectorGlobal.z = 0.0;

    // Estimate cornering stiffness for front and rear tires
    double frontLeftLatStiff = attributes.wheelPhysics.frontLeft[6];
    double frontRightLatStiff = attributes.wheelPhysics.frontRight[6];
    double rearLeftLatStiff = attributes.wheelPhysics.rearLeft[6];
    double rearRightLatStiff = attributes.wheelPhysics.rearRight[6];
    egoParams.Cf = (frontLeftLatStiff + frontRightLatStiff) / 2.0;
    egoParams.Cr = (rearLeftLatStiff + rearRightLatStiff) / 2.0; 

    // Calculate the distance from the center of mass to the front axle in meters
    static double _Lf = -1; // For now at spawn x=-6.446170, y=-79.055023, z=0.275307
    if (_Lf == -1) {
        const double frontAxleX_cm = (attributes.wheelPhysics.frontLeft[7] + attributes.wheelPhysics.frontRight[7]) / 2;
        const double frontAxleY_cm = (attributes.wheelPhysics.frontLeft[8] + attributes.wheelPhysics.frontRight[8]) / 2;
        const double frontAxleZ_cm = (attributes.wheelPhysics.frontLeft[9] + attributes.wheelPhysics.frontRight[9]) / 2;
        const double frontAxleX = (frontAxleX_cm / 100) - (-6.446170);
        const double frontAxleY = (frontAxleY_cm / 100) - (-79.055023);
        const double frontAxleZ = (frontAxleZ_cm / 100) - (0.275307);
        _Lf = sqrt(pow(frontAxleX - attributes.centerOfMass.x, 2) + 
                pow(frontAxleY - attributes.centerOfMass.y, 2) + 
                pow(frontAxleZ - attributes.centerOfMass.z, 2));
    }
    egoParams.Lf = _Lf;

    // Calculate the distance from the center of mass to the rear axle
    static double _Lr = -1; // For now at spawn x=-6.446170, y=-79.055023, z=0.275307
    if (_Lr == -1) {
        const double rearAxleX_cm = (attributes.wheelPhysics.rearLeft[7] + attributes.wheelPhysics.rearRight[7]) / 2;
        const double rearAxleY_cm = (attributes.wheelPhysics.rearLeft[8] + attributes.wheelPhysics.rearRight[8]) / 2;
        const double rearAxleZ_cm = (attributes.wheelPhysics.rearLeft[9] + attributes.wheelPhysics.rearRight[9]) / 2;
        const double rearAxleX = (rearAxleX_cm / 100) - (-6.446170);
        const double rearAxleY = (rearAxleY_cm / 100) - (-79.055023);
        const double rearAxleZ = (rearAxleZ_cm / 100) - (0.275307);
        _Lr = sqrt(pow(rearAxleX - attributes.centerOfMass.x, 2) + 
                pow(rearAxleY - attributes.centerOfMass.y, 2) + 
                pow(rearAxleZ - attributes.centerOfMass.z, 2));
    }
    egoParams.Lr = _Lr;

    //Moment of intertia rectangular box approx -> 0.95 M*(L/2)^2 where L = wheelbase
    egoParams.Iz = 0.95 * egoParams.m * pow((egoParams.Lf + egoParams.Lr) / 2, 2); 
    
    // Sideslip angle ->_beta = atan2(vyLocal, vxLocal);
    double _beta = atan(egoParams.Lr / (egoParams.Lf + egoParams.Lr) * tan(currentEgoControl.steer)); 

    // Velocity in m/s
    currentEgoState.vx = velocity.x * cos(currentEgoState.psi) + velocity.y * sin(currentEgoState.psi);
    currentEgoState.vx = currentEgoState.vx * cos(_beta) - currentEgoState.vx * sin(_beta);
    currentEgoState.vy = -velocity.x * sin(currentEgoState.psi) + velocity.y * cos(currentEgoState.psi);
    currentEgoState.vy = currentEgoState.vy * sin(_beta) + currentEgoState.vy * cos(_beta); // vx * tan(_beta)

    // Forward speed in m/s (kph*3.6)
    currentEgoState.vx = sqrt(pow(velocity.x, 2) + pow(velocity.y, 2) + pow(velocity.z, 2));

    // Instantiate ego output processing
    static size_t _fiter = 0;
    if(_fiter) {
        auto start = chrono::high_resolution_clock::now();

        motionControl();

        auto end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "execrt: " << elapsed.count() << "ms\n";
    }
    _fiter++;

    plotData.xEgo.push_back(currentEgoState.x);
    plotData.yEgo.push_back(currentEgoState.y);
    plotData.yaw.push_back(currentEgoState.psi);
    plotData.vx.push_back(currentEgoState.vx * 3.6);
    plotData.cte.push_back(currentEgoControl.cte);
    plotData.epsi.push_back(currentEgoControl.epsi);
    plotData.steer.push_back(currentEgoControl.steer);
    plotData.app.push_back(currentEgoControl.throttle);
    plotData.brake.push_back(currentEgoControl.brake);
    plotData.mpcost.push_back(currentEgoControl.mpcost);
    plotData.vError.push_back(REF_V);
}
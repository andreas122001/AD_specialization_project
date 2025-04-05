#!/usr/bin/env bash

cd ..
mkdir data
cd data

down_load_unzip() {
  wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset/$1.tar
  tar -xf $1.tar
  rm $1.tar
}

# Download 2024 garage_v1 (lb2) dataset

# Note, this will probably fail if you can't fit 2x the whole dataset on your disk. The tarfiles are untared and deleted in parallel, so worst case, all tarfiles are 100% untared at the same time before all of them are deleted. Make it sequential if you'd like.
for scenario in Accident AccidentTwoWays BlockedIntersection ConstructionObstacle ConstructionObstacleTwoWays ControlLoss CrossingBicycleFlow DynamicObjectCrossing EnterActorFlow EnterActorFlowV2 HardBreakRoute HazardAtSideLane HazardAtSideLaneTwoWays HighwayCutIn HighwayExit InterurbanActorFlow InterurbanAdvancedActorFlow InvadingTurn MergerIntoSlowTraffic MergerIntoSlowTrafficV2 NonSignalizedJunctionLeftTurn NonSignalizedJunctionRightTurn noScenarios OppositeVehicleRunningRedLight OppositeVehicleTakingPriority ParkedObstacle ParkedObstacleTwoWays ParkingCrossingPedestrian ParkingCutIn ParkingExit PedestrianCrossing PriorityAtJunction SignalizedJunctionLeftTurn SignalizedJunctionRightTurn StaticCutIn VehicleOpensDoorTwoWays VehicleTurningRoute VehicleTurningRoutePedestrian YieldToEmergencyVehicle
do
  down_load_unzip "${scenario}" &
done


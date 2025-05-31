cd /cluster/work/andrebw/repos/temporal_garage/Bench2Drive

clean_carla="/cluster/work/andrebw/repos/temporal_garage/Bench2Drive/tools/clean_carla.sh"


FOLDERS=(
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/static-LB5s[0-9]_e30/static-LB5s[0-9]_r[0-9]/results/bench2drive_split
)

i=0
for FOLDER in ${FOLDERS[@]}; do
    i=$((i+1))
    echo "Processing folder ($i/${#FOLDERS[@]}): $FOLDER"
    if test -f $FOLDER/merged_ability.json; then
        echo "Already processed. Skipping..."
        continue
    fi

    # Merge the results
    echo "Merging results..."
    python tools/merge_route_json.py -f $FOLDER > /dev/null
    echo Merged! Exit code: $?

    echo "Calculating multi-ability metrics..."
    sh $clean_carla
    # Calculate the multi-ability metrics
    until python tools/ability_benchmark.py -r $FOLDER/merged.json -p 10666 > /dev/null; do
        echo "Seems like carla failed: exit code $?. retrying..."
        echo "Cleaning up Carla before continuing..."
        sh $clean_carla
        sleep 1
    done
done

echo "Completed!"

cd /cluster/work/andrebw/repos/temporal_garage/Bench2Drive

clean_carla="/cluster/work/andrebw/repos/temporal_garage/Bench2Drive/tools/clean_carla.sh"


FOLDERS=(
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/tfpp_default_e30/tfpp_default_r[0-9]/results/bench2drive_split
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/static-LB[0-9]s[0-9]*_e30/static-LB[0-9]s[0-9]*_r[0-9]/results/bench2drive_split
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/LB5s1_e30/LB5s1_r[0-9]/results/bench2drive_split
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/lidar-LB5s1_e30/lidar-LB5s1_r[0-9]/results/bench2drive_split
    /cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/larger-LB9s2_e30/larger-LB9s2_r[0-9]/results/bench2drive_split
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
    n=0
    max_retry=9
    # Calculate the multi-ability metrics
    until python tools/ability_benchmark.py -r $FOLDER/merged.json -p 10666 > /dev/null || [ "$n" -ge "$max_retry" ]; do
        echo "Seems like carla failed: exit code $?. retrying... ($((n+1)) / $((max_retry+1)))"
        sh $clean_carla
        sleep 1
    done
done

echo "Completed!"

cd /cluster/work/andrebw/repos/temporal_garage/Bench2Drive

clean_carla="/cluster/work/andrebw/repos/temporal_garage/Bench2Drive/tools/clean_carla.sh"

if [ "$#" -ne 1 ]; then VERSION=v3; else VERSION=$1; fi;


FOLDERS=(
    /cluster/work/andrebw/repos/temporal_garage/evaluation/$VERSION/bench2drive/**/*/results/bench2drive_split
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

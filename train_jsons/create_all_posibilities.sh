# We have three values to configure the model:
# - `cnn`: can be either `resnet18` or `resnet50` or `simple`
# - `loss-margin`: a float value that determines the margin for the contrastive loss we will keep it at 1.0 for now
# - `distance`: can be either `euclidean` or `cosine`

for model in resnet18 resnet50 simple; do
    for distance in euclidean cosine; do
        printf "Configuring model %s with distance %s\n" "$model" "$distance"
        printf "{\"cnn\": \"%s\", \"loss-margin\": 1.0, \"distance\": \"%s\"}\n" "$model" "$distance" >> config_${model}_${distance}.json
    done
done
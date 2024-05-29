import visualize
import predict
from pathlib import Path
import project
def main(image_dir, predicted_dir, visualized_dir):
    # predict.main(image_dir, predicted_dir)
    visualize.main(image_dir, predicted_dir, visualized_dir)

if __name__=="__main__":
    root_dir = Path(project.PROJECT_ROOT)/"origin_datasets/sangmu_real_image_det(cropped)"
    # root_dir = Path(project.PROJECT_ROOT)/"origin_datasets/test"
    image_dir = root_dir/"images"
    predicted_dir = root_dir/"predicted"
    visualized_dir = root_dir/"visualized"

    predicted_dir.mkdir(parents=True, exist_ok=True)
    visualized_dir.mkdir(parents=True, exist_ok=True)

    main(image_dir, predicted_dir, visualized_dir)
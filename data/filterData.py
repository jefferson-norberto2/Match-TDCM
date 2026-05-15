import csv
from pycocotools.coco import COCO
import math

def filter_data(coco_annotations_path, output_csv_path, tShape=(36, 36), scale_range=(0.5, 2)):
    """
    Filters targets in the COCO dataset that meet the scaling conditions for the original 
    width and height (judged directly based on the original size).
      
    Args:
        coco_annotations_path: Path to the COCO dataset annotation file (e.g., instances_train2017.json)
        output_csv_path: Path to the output CSV file
        tShape: Target size of the template, defaults to (36, 36)
        scale_range: Scaling range of w and h relative to tShape, defaults to (0.5, 2)
    """
    th, tw = tShape
    count = 0
    
    # Load the COCO dataset
    coco = COCO(coco_annotations_path)
    img_ids = coco.getImgIds()
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # csv_writer.writerow(['img_name', 'x', 'y', 'w', 'h'])  # Write header
        
        # Iterate through all images
        for img_id in img_ids:
            # Get image information
            img_info = coco.loadImgs(img_id)[0]
            img_name = img_info['file_name']
            
            # Get annotations corresponding to the image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # Iterate through all annotations in the image
            for ann in anns:
                # Get bounding box [x, y, width, height]
                bbox = ann['bbox']  
                x, y, w, h = bbox
                x = math.floor(x)
                y = math.floor(y)
                w = math.ceil(w)
                h = math.ceil(h)
                
                # Calculate the scaling ratio of w and h relative to the target size
                scale_w = w / tw
                scale_h = h / th
                
                # Check if scaling conditions are met
                if (scale_range[0] <= scale_w <= scale_range[1] and 
                    scale_range[0] <= scale_h <= scale_range[1]):
                    
                    # If it needs to be scaled to 1 and there is not enough data, cropping is required
                    # w, h = tw, th
                    
                    # Write data meeting the conditions to the CSV
                    csv_writer.writerow([
                        img_name,
                        x, y, w, h
                    ])
                    count += 1
                    
    print(f"Filtering complete, found {count} targets meeting the conditions.")


if __name__ == "__main__":
    # COCO dataset annotation file path
    annotations_path = "/path-to/MS-CoCo/annotations/instances_train2017.json"
    
    # Output CSV file path
    output_path = 'train.csv'
    
    filter_data(
        annotations_path, 
        output_path, 
        tShape=(36, 36), 
        scale_range=(0.5, 2)
    )
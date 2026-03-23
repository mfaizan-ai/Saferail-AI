import cv2
import numpy as np

def masks_to_points_per_image(masks_per_image):
    masks_per_image = masks_per_image.astype(np.uint8)
    if masks_per_image.max() <= 1:
        masks_per_image *= 255
    image_masks_pts = {}
    for i, mask in enumerate(masks_per_image):
        image_masks_pts[f'mask{i+1}'] = np.argwhere(mask == 255)
    return image_masks_pts

def mask_to_points_per_batch(masks_per_batch):
    per_batch_pts = []
    for img_masks in masks_per_batch:
        img_masks_points = masks_to_points_per_image(img_masks)
        per_batch_pts.append(img_masks_points)
    return per_batch_pts


def sort_points(points):
    return points[np.argsort(points[:, 1])]

def create_occupancy_region(track1_points, track2_points):
    track1_sorted = sort_points(track1_points)
    track2_sorted = sort_points(track2_points)
    return np.concatenate((track1_sorted, track2_sorted[::-1]), axis=0)

def is_person_inside(polygon, point):
    return cv2.pointPolygonTest(polygon.astype(np.int32), point, False) >= 0


if __name__ == '__main__':
    import random
    random.seed(444)

    per_batch_masks = np.random.choice([0, 1], size=(2, 2, 4, 4))
    print('per_batch_masks: ')
    print(per_batch_masks)
    regions = []
    per_batch_points = mask_to_points_per_batch(per_batch_masks)
    for per_image_pts in per_batch_points:
        occupied_region = create_occupancy_region(per_image_pts['mask1'], per_image_pts['mask2'])
        regions.append(occupied_region)
    

         
        
    





# def detect_intrusion(masks):
#     '''
#     Detect intrusion
#     ----------------
#     '''

    


# # track_masks: tensor with 2 binary masks (2, H, W)
# mask1, mask2 = track_masks[0], track_masks[1]

# # Get track points
# track1_points = mask_to_points(mask1)
# track2_points = mask_to_points(mask2)

# # Create polygon
# occupancy_polygon = create_occupancy_region(track1_points, track2_points)

# # For each detected person:
# for person_bbox in person_detections:  # e.g., [x1, y1, x2, y2]
#     cx = int((person_bbox[0] + person_bbox[2]) / 2)
#     cy = int(person_bbox[3])  # use foot point
#     if is_person_inside(occupancy_polygon, (cx, cy)):
#         print("⚠️ Intrusion Detected!")

import numpy as np
from PIL import Image, ImageDraw

def crop_by_polygon(image, polygon):
    polygon = sum(polygon, [])
    """_summary_
    iamge를 polygon 영역에 대해 crop 해 준다.

    Args:
        image (PIL.Image): _description_
        polygon (list): [(x1, y1), (x2, y2), ... , (xn, yn)]

    Returns:
        _type_: polygon_points 영역에 대해 cropped된 image 객체(PIL.Image)
    """

    # Create a mask image
    mask = Image.new('L', image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

    # Convert the mask to a NumPy array
    mask_np = np.array(mask)

    # Apply the mask to the image
    image_np = np.array(image)
    result = np.zeros_like(image_np)
    result[mask_np == 1] = image_np[mask_np == 1]

    # Create a new image from the result array
    result_image = Image.fromarray(result)

    # Find bounding box of the polygon
    bbox = mask.getbbox()

    # Crop the image to the bounding box
    cropped_image = result_image.crop(bbox)

    return cropped_image

def is_point_inside_polygon(point, polygon):
    """_summary_

    Args:
        point (list): 내부에 있는지 확인할 point (x, y)
        polygon (list): 영역을 나타내는 polygon 영역 [(x1, y1), (x2, y2), ... , (xn, yn)]

    Returns:
        bool: point가 polygon 내부에 위치하는지 여부
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def is_polygon_inside_polygon(polygon1, polygon2):    # polygon1이 polygon2에 완전히 포함되는지
    """_summary_

    Args:
        polygon1 (list): 상대 polygon에 포함되는지 확인할 polygon [(x1, y1), (x2, y2), ... , (xn, yn)]
        polygon2 (list): 상대 polygon을 감싸는지 체크할 polygon [(x1, y1), (x2, y2), ... , (xn, yn)]

    Returns:
        bool: polygon1이 polygon2에 포함되는지 여부
    """
    return all([is_point_inside_polygon(point, polygon2)  for point in polygon1])
    
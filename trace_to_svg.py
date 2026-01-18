import cv2
import numpy as np

# --- CONFIGURATION ---
REAL_WORLD_WIDTH_MM = 150.0 
MARGIN_MM = 3.0  # The border to ignore

def generate_svg_from_contoursOLD(binary_image, output_path):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    width_mm = REAL_WORLD_WIDTH_MM
    svg_header = f'<svg width="{width_mm}mm" height="{width_mm}mm" viewBox="0 0 {binary_image.shape[1]} {binary_image.shape[0]}" xmlns="http://www.w3.org/2000/svg">\n'
    
    paths = []
    for cnt in contours:
        if len(cnt) < 3: continue 
        epsilon = 1.5 
        approx = cv2.approxPolyDP(cnt, epsilon, False)

        path_data = "M " + " L ".join([f"{p[0][0]},{p[0][1]}" for p in approx]) + " Z"
        # Changed fill to "none" and added a stroke
        paths.append(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="1" />')
    
    with open(output_path, "w") as f:
        f.write(svg_header + "\n".join(paths) + "\n</svg>")

def generate_svg_from_contours(binary_image, output_path):
    # Use RETR_CCOMP to get a 2-level hierarchy (Outer boundaries vs holes)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None: return

    width_mm = REAL_WORLD_WIDTH_MM
    svg_header = f'<svg width="{width_mm}mm" height="{width_mm}mm" viewBox="0 0 {binary_image.shape[1]} {binary_image.shape[0]}" xmlns="http://www.w3.org/2000/svg">\n'
    
    paths = []
    # We use fill-rule="evenodd" so that holes (like eyes) stay white
    svg_path_start = '  <path fill-rule="evenodd" fill="black" d="'
    path_data_segments = []

    for i, cnt in enumerate(contours):
        # 1. Ignore the giant outer box if it exists (contours that match image size)
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= binary_image.shape[1] - 5 and h >= binary_image.shape[0] - 5:
            continue
            
        # 2. Ignore tiny noise
        if cv2.contourArea(cnt) < 10:
            continue

        # 3. Smooth the lines
        epsilon = 1.0 
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 4. Build the path string
        segment = "M " + " L ".join([f"{p[0][0]},{p[0][1]}" for p in approx]) + " Z "
        path_data_segments.append(segment)

    if not path_data_segments:
        print("No paths found!")
        return

    full_path = svg_path_start + "".join(path_data_segments) + '" />'
    
    with open(output_path, "w") as f:
        f.write(svg_header + full_path + "\n</svg>")
def filter_small_parts(binary_image, min_area=50):
    # Find all parts
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank black mask
    mask = np.zeros_like(binary_image)
    
    for cnt in contours:
        # If the part is large enough, draw it onto our mask
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask, [cnt], -1, 255, -1) # -1 means fill the shape
            
    return mask
def process_frame(image_path, output_svg="output.svg"):
    img = cv2.imread(image_path)
    if img is None: return print("File not found")

    # Detect ArUco Markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, rejected = detector.detectMarkers(img)

    # cv2.aruco.drawDetectedMarkers(img,corners,ids,(255,0,0))
    

    if ids is None or len(ids) < 4:
        print(f"Error: Need 4 markers, found {len(ids) if ids is not None else 0}")
        return f"Error: Need 4 markers, found {len(ids) if ids is not None else 0}"

    id_to_inner_idx = {0: 2, 1: 3, 2: 0, 3: 1}
    points_map = {marker_id: corners[i][0][id_to_inner_idx[marker_id]] 
                  for i, marker_id in enumerate(ids.flatten()) if marker_id in id_to_inner_idx}

    if len(points_map) < 4: 
        print("Missing Marker IDs")
        return ("Missing Marker IDs")

    # Warp Perspective
    src_pts = np.array([points_map[0], points_map[1], points_map[2], points_map[3]], dtype="float32")
    output_px = int(REAL_WORLD_WIDTH_MM * 10)
    dst_pts = np.array([[0,0], [output_px,0], [output_px,output_px], [0,output_px]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (output_px, output_px))

    # Thresholding
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(warped_gray)
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
    contrast_enhanced = clahe.apply(blurred)
    smoothed = cv2.medianBlur(contrast_enhanced, 5)
    # binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY_INV, 15, 5)
    binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 41, 17)
    kernel_noise = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    # B. Bridge the "Steps"
    # Dilation connects the broken pixels into a solid "blob" line
    binary = cv2.dilate(binary, kernel_noise, iterations=2)

    # C. Ignore Outer 3mm
    pixel_margin = int(5.0 * 10) 
    mask = np.zeros_like(binary)
    cv2.rectangle(mask, (pixel_margin, pixel_margin), 
                  (output_px - pixel_margin, output_px - pixel_margin), 255, -1)
    binary = cv2.bitwise_and(binary, mask)
    # binary = filter_small_parts(binary, min_area=2)
    # D. Thinning (Collapsing the blob into a spine)
    # This removes the double-line effect after we bridged the steps
    try:
        skeleton = cv2.ximgproc.thinning(binary)
    except:
        # Fallback to basic erosion if ximgproc isn't available
        skeleton = cv2.erode(binary, kernel_noise, iterations=1)
    
    # E. Generate SVG (using the updated function above)
    cv2.imshow("Binary",binary)
    # cv2.waitKey(0)
    generate_svg_from_contours(binary, output_svg)
    # kernel = np.ones((3,3), np.uint8)
    # binary = cv2.dilate(binary, kernel, iterations=1)
    
    # # 1. IGNORE OUTER 3MM
    # # Since 150mm = 1500px (from your output_px calculation), 3mm = 30px
    # pixel_margin = int(3.0 * 10) 
    # mask = np.zeros_like(binary)
    # # Create a white rectangle inside the black mask to keep only the center
    # cv2.rectangle(mask, 
    #               (pixel_margin, pixel_margin), 
    #               (output_px - pixel_margin, output_px - pixel_margin), 
    #               255, -1)
    # binary = cv2.bitwise_and(binary, mask)


    # # 5. ROBUST CENTER-LINE EXTRACTION (Skeletonization)
    # # Now that the line is smooth and solid, thinning will produce a clean path
    # dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # _, skeleton = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    # skeleton = np.uint8(skeleton)
    
    # Final thinning to ensure it's strictly 1-pixel wide
    # skeleton = cv2.ximgproc.thinning(skeleton)

    # # 2. FIX HOLLOW LINES (Skeletonization)
    # # This reduces the thick pen strokes to a 1-pixel wide line
    # # Note: Requires 'pip install opencv-contrib-python'
    # try:
    #     raise AttributeError()
    #     # binary = cv2.ximgproc.thinning(binary)
    # except AttributeError:
    #     # Fallback if ximgproc is not installed: use morphological thinning
    #     kernel = np.ones((3,3), np.uint8)
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
    #     print("Warning: ximgproc not found. Using basic erosion instead.")

    # 3. GENERATE THE SVG
    # generate_svg_from_contours(skeleton, output_svg)
    # print(f"Success! {output_svg} created.")
    # print(f"Success! {output_svg} created (3mm border ignored).")
    return True

# process_frame("test_a.jpg")
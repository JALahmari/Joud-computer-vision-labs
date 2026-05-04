import cv2
import time
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="../media/skydiving.mp4", help="Input Video")
parser.add_argument("--mode", default="MPI", choices=["COCO", "MPI"], help="OpenPose model to use")
parser.add_argument("--output", default=None, help="Path for the output video")
parser.add_argument("--show", action="store_true", help="Display frames while processing")

args = parser.parse_args()

SCRIPT_DIR = Path(".") if (Path.cwd() / "mpi").exists() else Path(__file__).parent


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


MODE = args.mode

if MODE == "COCO":
    protoFile = SCRIPT_DIR / "coco" / "pose_deploy_linevec.prototxt"
    weightsFile = SCRIPT_DIR / "coco" / "pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI" :
    protoFile = SCRIPT_DIR / "mpi" / "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = SCRIPT_DIR / "mpi" / "pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.1


input_source = resolve_path(args.video_file)
if not input_source.exists():
    raise FileNotFoundError(f"Input video not found: {input_source}")
if not protoFile.exists():
    raise FileNotFoundError(f"OpenPose prototxt file not found: {protoFile}")
if not weightsFile.exists():
    raise FileNotFoundError(
        f"OpenPose weights file not found: {weightsFile}\n"
        "Download the required .caffemodel from README.md and place it in the matching model folder."
    )

cap = cv2.VideoCapture(str(input_source))
if not cap.isOpened():
    raise RuntimeError(f"Unable to open video: {input_source}")

hasFrame, frame = cap.read()
if not hasFrame:
    raise RuntimeError(f"Unable to read the first frame from: {input_source}")

save_name = input_source.stem
output_path = Path(args.output) if args.output else SCRIPT_DIR / "outputs" / f"{save_name}_openpose.avi"
if not output_path.is_absolute():
    output_path = SCRIPT_DIR / output_path
output_path.parent.mkdir(parents=True, exist_ok=True)
video_fps = int(cap.get(cv2.CAP_PROP_FPS))
vid_writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*'MJPG'),
    video_fps if video_fps > 0 else 10,
    (frame.shape[1],frame.shape[0]),
)
if not vid_writer.isOpened():
    raise RuntimeError(f"Unable to create output video: {output_path}")

net = cv2.dnn.readNetFromCaffe(str(protoFile), str(weightsFile))
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

frame_count = 0

while True:
    if not hasFrame:
        break

    t = time.time()
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)

    vid_writer.write(frame)
    frame_count += 1

    if args.show:
        cv2.imshow('Output-Skeleton', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    hasFrame, frame = cap.read()

cap.release()
vid_writer.release()
cv2.destroyAllWindows()
print(f"Processed {frame_count} frames")
print(f"Saved OpenPose output to: {output_path}")

import time
import torch
import cv2
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    # Letterbox resizing
    img = letterbox(frame, input_size, stride=64, auto=True)[0]

    # Convert to tensor
    img = transforms.ToTensor()(img)
    img = torch.tensor(np.array([img.numpy()])).to(device)

    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()
        fps = 1 / (t2 - t1)

        output = non_max_suppression_kpt(
            output,
            0.25,   # confidence threshold
            0.65,   # IoU threshold
            nc=1,
            nkpt=17,
            kpt_label=True
        )

        output = output_to_keypoint(output)

    # Convert image back for display
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    # Draw skeleton
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, fps


# ================== SETTINGS ==================

input_size = 256

# USE YOUR VIDEO HERE
vid_path = r"C:\Users\djood\OneDrive\المستندات\GitHub\Joud-computer-vision-labs\lab08-pose-estimation\media\skydiving.mp4"
save_name = "skydiving"

# ============================================

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected Device:", device)

# Load model
weights = torch.load('yolov7-w6-pose.pt', map_location=device, weights_only=False)
model = weights['model']
model.float().eval().to(device)

# Load video
cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
ret, frame = cap.read()

if not ret:
    print("Error: Cannot read video.")
    exit()

h, w, _ = frame.shape

# Video writer
out = cv2.VideoWriter(
    f"{save_name}_yolo7.avi",
    cv2.VideoWriter_fourcc(*'MJPG'),
    fps,
    (w, h)
)

# ================== MAIN LOOP ==================

if __name__ == '__main__':
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Finished processing.")
            break

        img, fps_ = pose_video(frame)

        cv2.putText(img, f'FPS: {fps_:.2f}', (200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'YOLOv7 Pose', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Output', img)

        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

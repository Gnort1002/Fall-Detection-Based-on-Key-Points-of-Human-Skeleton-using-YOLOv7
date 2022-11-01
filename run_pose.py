import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
from PIL import Image, ImageDraw
import telepot
import os

token = '5488308920:AAHmY9R_zUwvolCq906cHdfjfgJ4eDPPymo'
receiver_id = '-1001854852651'
bot = telepot.Bot(token)

@torch.no_grad()
def run(poseweights= 'yolov7-w6-pose.pt', source='pose.mp4', device='cpu'):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        vid_write_image = letterbox(
            cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))

        frame_count, total_fps = 0, 0
        human_fall_detected = []
        time_start_system = time.time()
        print(time_start_system)
        time_start_falling = 0
        time_now = 0
        while cap.isOpened:
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                icon = cv2.imread("wet-floor.png")

                thre = (frame_height//2) + 100
                
                for idx in range(output.shape[0]):
                    kpts = output[idx, 7:].T
                    #plot_skeleton_kpts(img, kpts, 3)
                    xmin, ymin = (output[idx, 2] - output[idx, 4]/2), (output[idx, 3] - output[idx, 5]/2)
                    xmax, ymax = (output[idx, 2] + output[idx, 4]/2), (output[idx, 3] + output[idx, 5]/2)
                    pt1 = (int(xmin), int(ymin))
                    pt2 = (int(xmax), int(ymax))
                    dx = (int(xmax)- int(xmin))
                    dy = (int(ymax)- int(ymin))
                    cx = (int(xmax + xmin)//2)
                    cy = (int(ymax + ymin)//2)
                    icon = cv2.resize(icon, (50, 50), interpolation=cv2.INTER_LINEAR)
                    difference = dy - dx
                    #0: head, 1: chest, 2/5: r/l shoulder, 3/6: r/l elbow, 4/8: r/l hand, 10/13: r/l foot
                    chest = Get_coord(kpts, 1, "y")
                    if ((difference > 0) and (int(chest) > thre)) or (difference < 0):
                        human_fall_detected.append(1)
                        draw_border(img, pt1, pt2, (84, 61, 247), 10)
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        #draw.rounded_rectangle((cx-10, cy-10, cx+60, cy+60), fill=(84, 61, 247), radius = 15)
                        draw.rounded_rectangle((cx-10, cy-10, cx+60, cy+60), fill=(84, 61, 247), radius = 15)
                        img = np.array(im)
                        if dx >= 100 and dy >= 100: img[cy:cy+50, cx:cx+50] =icon
                        if time_start_falling == 0: 
                            time_start_falling = time.time()
                            
                    else: 
                        time_start_falling = 0
                        time_now = 0
                    if (time_start_falling != 0):
                        time_now = time.time()
                        if (time_now - time_start_falling) > 10:
                            bot.sendMessage(receiver_id, "Person Fall Detected")                    
                            filename = r"falled_img\savedImage.jpg"
                            cv2.imwrite(filename, img)
                            bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                            os.remove(filename)
                            time_start_falling = 0
                            time_now = 0
                img_ = img.copy()
                img_ = cv2.resize(
                    img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Detection", img_)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    break

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                print(f"Frame {frame_count} Processing. Frames Per Second : {fps}")
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    opt = parser.parse_args()
    return opt

def Get_coord(kpts, id_part, coord):
    if coord == "x": return kpts[3 * id_part]
    if coord == "y": return kpts[3 * id_part + 1]
    if coord == "xy": return kpts[3 * id_part], kpts[3 * id_part + 1]

def draw_border(img, pt1, pt2, color, thickness):
    cv2.rectangle(img, pt1, pt2, color, thickness = thickness)

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)

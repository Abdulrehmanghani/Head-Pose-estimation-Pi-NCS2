import sys
import cv2
import time
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder
import headvisualizer as hvis
import altusi.visualizer as vis

from altusi.facedetector import FaceDetector
from altusi.headposer import HeadPoser


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()                        
   
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="CAM or path to image or video file.")
    
    parser.add_argument("-disp", "--display", required=False, default=True, type=str,
                        help="Flag to display the outputs of the intermediate models")

    parser.add_argument("-d", "--device", required=False, default="CPU", type=str,
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD.")
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
                       
    return parser

def handle_input_type(input_stream):
    '''
     Handle image, video or webcam
    '''
    
    # Check if the input is an image
    if input_stream.endswith('.jpg') or input_stream.endswith('.png') or input_stream.endswith('.bmp'):
        input_type = 'image'
        
    # Check if the input is a webcam
    elif input_stream == 'CAM':
        input_type = 'cam'
        
    # Check if the input is a video    
    elif input_stream.endswith('.mp4'):
        input_type = 'video'
    else: 
        log.error('Please enter a valid input! .jpg, .png, .bmp, .mp4, CAM')    
        sys.exit()    
    return input_type

def infer_on_stream(args):
    """
    Initialize the inference networks, stream video to network,
    and output stats, video and control the mouse pointer.
    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
    # Initialise the classes
    # Load the models 
    start_load = time.time()
    try:
        detector = FaceDetector()
        plugin = detector.getPlugin()
        poser = HeadPoser(plugin=plugin)
    except:
        log.error('Please enter a valid model file address')    
        sys.exit()
    
    log.debug("Models loaded: time: {:.3f} ms".format((time.time() - start_load) * 1000))
    end_load = time.time() -  start_load 
    
    # Handle the input stream
    input_type = handle_input_type(args.input)
    
    # Initialise the InputFeeder class
    feed = InputFeeder(input_type=input_type, input_file=args.input)
    
    # Load the video capture
    feed.load_data()
    frame_count = 0
    start_inf = time.time()
    
    # Read from the video capture 
    for flag, frame in feed.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame_count += 1
        try:
            # Run inference on the models     
            scores, bboxes = detector.getFaces(frame)
            ## If no face detected move back to the top of the loop
            if len(bboxes):
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    face_image = frame[y1:y2, x1:x2]
                    yaw, pitch, roll = poser.estimatePose(face_image)

                    cpoint = [(x1+x2)//2, (y1+y2)//2]
                    frame = hvis.draw(frame, cpoint, (yaw, pitch, roll))
            _prx_t = time.time() - start_inf

            #frame = vis.plotInfo(frame, 'Raspberry Pi - FPS: {:.3f}'.format(1/_prx_t))
            #frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
            if args.display:
                cv2.imshow('Computer pointer control', cv2.resize(frame,(600,400)))
            #mouse_controller.move(gaze_vector[0], gaze_vector[1])
        except Exception as e:
            log.warning(str(e) + " for frame " + str(frame_count))
            continue
        # Display the resulting frame
        
     
    end_inf = time.time() - start_inf
    log.info("\nTotal loading time: {}\nTotal inference time: {}\nFPS: {}".format(end_load, end_inf,frame_count/end_inf))
    
    # Release the capture
    feed.close()
    # Destroy any OpenCV windows
    cv2.destroyAllWindows
    log.debug("The program debug successfully")

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    #time.sleep(7)
    # Grab command line args
    args = build_argparser().parse_args()

    #Perform inference on the input stream
    infer_on_stream(args)

if __name__ == '__main__':
    main()
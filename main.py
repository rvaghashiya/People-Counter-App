"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CLASSES = ['person']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser
   
def draw_boxes(frame, result, prob_threshold, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    pcnt=0
    centres=[]
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
            #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
            pcnt+=1
            centres.append([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
            #cv2.circle(frame, (int(centres[-1][0]), int(centres[-1][1])), 3, (0,255,255), -1 )
        #cv2.line(frame, (width-150,0),(width-150,height), (255,255,0), 2)

    return frame, pcnt, centres

def person_inframe(curr_centre, prev_centre, width, height, margin=30):
    new=len(curr_centre)
    old=len(prev_centre)

    p_inframe,p_transit = -3,-3

    if old==0 and new==1:
        #new person entry
        p_inframe=1
        p_transit=1

    elif old==new and new==1:
        #detected person is in frame
        p_inframe =1
        p_transit =0

    elif old==1 and new==0:
        #either missed detection or person left
        #assume: if y is near border, person left else skipped detection
        if width-prev_centre[0][0]>margin:
            #print("width",width, "prev centre",prev_centre,"diff", width-prev_centre[0][0])
            p_inframe=-1
            p_transit=0
        else:
            p_inframe=0
            p_transit=-1

    return p_inframe, p_transit


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam or img
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
    elif not args.input.endswith('.mp4'):
        print("incorrect source! unable to use input")
        exit(1)
    
    ### TODO: Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    tot_cnt  = 0
    curr_cnt = 0
    prev_centre = []
    person = False
    margin=150

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        inf_time = time.time()
        infer_network.exec_net(p_frame)
        
        ### TODO: Wait for the result ###
        # Get the output of inference
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            inf_time_dur = int( (time.time() - inf_time) *1000 )
            textt= "inference time : "+ str(inf_time_dur)+" ms"
            cv2.putText(frame, textt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            ### TODO: Update the frame to include detected bounding boxes and count persons in frame
            ### TODO: Extract any desired stats from the results ###
            ### TODO: Process the output
            frame, loc_cnt, centre = draw_boxes(frame, result, prob_threshold, width, height)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            p_inframe, p_transit = person_inframe(centre, prev_centre, width, height, margin)

            if p_inframe==1 and p_transit==1 and person==False: #person entered
                if width-centre[0][0]>margin:                    
                    person=True
                    prev_centre=centre
                    curr_cnt+=1
                    tot_cnt+=1
                    duration=0
                    #print("person entered","curr",curr_cnt,"tot",tot_cnt)
                    start_time = time.time()
                    #client.publish("person", json.dumps({"count": curr_cnt}))
                    client.publish("person", json.dumps({"count": curr_cnt, "total": tot_cnt}))
                else:
                    0
                    #ignore
                    #print("ignore, person was leaving", "curr",curr_cnt,"tot",tot_cnt)

            elif p_inframe==1 and p_transit==0: # person in scene
                #print("person still in scene","curr",curr_cnt,"tot",tot_cnt)
                prev_centre=centre
                curr_durr = int(time.time() - start_time)
                if curr_durr > 20 :
                    cv2.putText(frame, "person exceeded allotted time", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                

            elif p_inframe==-1 and p_transit==0: #missed detection
                #print("missed detection, person still there", "curr",curr_cnt,"tot",tot_cnt)
                #print("centre",centre,"prevcentre",prev_centre)
                prev_centre = prev_centre
                curr_durr = int(time.time() - start_time)
                if curr_durr > 20 :
                    cv2.putText(frame, "person exceeded allotted time", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

            elif p_inframe==0 and p_transit==-1: #person left
                curr_cnt = curr_cnt-1
                #print("person left", "curr",curr_cnt,"tot",tot_cnt)
                #print("centre",centre,"prevcentre",prev_centre)
                prev_centre=[]
                person=False

                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
                #client.publish("person", json.dumps({"count": curr_cnt}))
            
            client.publish("person", json.dumps({"count": curr_cnt}))
            #client.publish("person", json.dumps({"total": tot_cnt}))
            # Person duration in the video is calculated

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        ### TODO: Write out the frame, depending on image or video     
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    ### TODO: Close the stream and any windows at the end of the application
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

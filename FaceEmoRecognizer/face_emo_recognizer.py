import  time
import cv2 as cv
import  argparse
import  psutil

# Import OpenVINO Inference Engine
from openvino.inference_engine import IECore, IENetwork


def run_app():
    frame_count = 0

    # Load Network
    OpenVinoNetwork = IENetwork(model=arguments.model_xml, weights=arguments.model_bin)

    # Get Input Layer Information
    InputLayer = next(iter(OpenVinoNetwork.inputs))
    print("Input Layer: ", InputLayer)

    # Get Output Layer Information
    OutputLayer = next(iter(OpenVinoNetwork.outputs))
    print("Output Layer: ", OutputLayer)

    # Get Input Shape of Model
    InputShape = OpenVinoNetwork.inputs[InputLayer].shape
    print("Input Shape: ", InputShape)

    # Get Output Shape of Model
    OutputShape = OpenVinoNetwork.outputs[OutputLayer].shape
    print("Output Shape: ", OutputShape)

    # Load IECore Object
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)

    # Load CPU Extensions if Necessary
    if 'CPU' in arguments.target_device:
        OpenVinoIE.add_extension('/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.so', "CPU")

    # Create Executable Network
    if arguments.async:
        print("Async Mode Enabled")
        OpenVinoExecutable = OpenVinoIE.load_network(network=OpenVinoNetwork, device_name=arguments.target_device, num_requests=number_of_async_req)
    else:
        OpenVinoExecutable = OpenVinoIE.load_network(network=OpenVinoNetwork, device_name=arguments.target_device)

    # Generate a Named Window to Show Output
    cv.namedWindow('Window', cv.WINDOW_NORMAL)
    cv.resizeWindow('Window', 800, 600)

    start_time = time.time()

    if arguments.input_type == 'image':
        frame_count += 1
        # Read Image
        image = cv.imread(arguments.input)

        # Get Shape Values
        N, C, H, W = OpenVinoNetwork.inputs[InputLayer].shape

        # Pre-process Image
        resized = cv.resize(image, (W, H))
        resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        input_image = resized.reshape((N, C, H, W))

        # Start Inference
        start = time.time()
        results = OpenVinoExecutable.infer(inputs={InputLayer: input_image})
        end = time.time()
        inf_time = end - start
        print('Inference Time: {} Seconds Single Image'.format(inf_time))

        fps = 1./(end-start)
        print('Estimated FPS: {} FPS Single Image'.format(fps))

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        text = 'FPS: {}, INF: {}'.format(round(fps, 2), round(inf_time, 2))
        cv.putText(image, text, (0, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        # Print Bounding Boxes on Image
        detections = results[OutputLayer][0][0]
        for  detection  in  detections :
            if detection[2] > arguments.detection_threshold:
                print('Original Frame Shape: ', fw, fh)
                xmin = int(detection[3] * fw)
                ymin = int(detection[4] * fh)
                xmax = int(detection[5] * fw)
                ymax = int(detection[6] * fh)
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
                text = '{}, %: {}'.format(int(detection[1]), round(detection[2], 2))
                cv.putText(image, text, (xmin, ymin - 7), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 125, 255), 1)

        cv.imshow('Window', image)
        cv.waitKey(0)
    else:
        print("No image.")
        return

    end_time = time.time()
    print('Elapsed Time: {} Seconds'.format(end_time - start_time))
    print('Number of Frames: {} '.format(frame_count))
    print('Estimated FPS: {}'.format(frame_count / (end_time - start_time)))


global  arguments
global number_of_async_req

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Basic OpenVINO Example with MobileNet-SSD')
    parser.add_argument('--model-xml',
                        default='./model/emotions-recognition-retail-0003.xml',
                        help='XML File')
    parser.add_argument('--model-bin',
                        default='./model/emotions-recognition-retail-0003.bin',
                        help='BIN File')
    parser.add_argument('--target-device', default='CPU',
                        help='Target Plugin: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU')
    parser.add_argument('--input-type', default='image', help='Type of Input: image, video, cam')
    parser.add_argument('--input', default='/home/intel/Pictures/cars.png',
                        help='Path to Input: WebCam: 0, Video File or Image file')

    parser.add_argument('--async',action="store_true", default=False, help='Run Async Mode')
    parser.add_argument('--request-number', default=1, help='Number of Requests')

    global  arguments
    arguments = parser.parse_args()

    global number_of_async_req
    number_of_async_req = int(arguments.request_number)

    run_app()
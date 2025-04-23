package org.firstinspires.ftc.teamcode;

import org.openftc.easyopencv.OpenCvPipeline;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

public class DetectionPipeline extends OpenCvPipeline {
    Mat hsv = new Mat();

    // Thresholding
    Scalar hsvLowerThreshold = new Scalar(80, 68, 0);
    Scalar hsvUpperThreshold = new Scalar(120, 255, 150);
    Mat thresholded = new Mat();
    Mat thresholdedRGB = new Mat();

    // Masking
    Mat masked = new Mat();
    Mat maskedRGB = new Mat();

    // Bounding box
    int[] bottomLeft;
    int[] topRight;

    // Output mat (with color labeling, object detection)
    Mat output = new Mat();

    @Override
    public Mat processFrame(Mat input) {
        // Convert to hsv
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        
        // HSV threshold
        Core.inRange(hsv, hsvLowerThreshold, hsvUpperThreshold, thresholded);

        // Mask original pixel values over only the thresholded values (display/debug only)
        Imgproc.cvtColor(thresholded, thresholdedRGB, Imgproc.COLOR_GRAY2RGBA);
        Core.bitwise_and(input, thresholdedRGB, masked);

        // Find bounding box
        // for (int i = 0; i < thresholded.rows(); i++) {
        //     for (int j = 0; j < thresholded.cols(); j++) {
        //         // If the pixel in thresholded
        //         if (thresholded.get(i, j)[0] == 255) {
        //             if (bottomLeft == null) {
        //                 bottomLeft[0] = i;
        //                 bottomLeft[1] = j;
        //             }
        //             if (topRight == null) {
        //                 topRight[0] = i;
        //                 topRight[1] = j;
        //             }
        //         }
        //     }
        // }

        return masked;
        // return output;
    }

}

package org.firstinspires.ftc.teamcode;

import org.openftc.easyopencv.OpenCvPipeline;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

public class DetectionPipeline extends OpenCvPipeline {
    Mat hsv = new Mat();

    // Thresholding
    Scalar hsvLowerThreshold = new Scalar(62, 67, 0);
    Scalar hsvUpperThreshold = new Scalar(155, 255, 150);
    Mat thresholded = new Mat();
    Mat thresholdedRGB = new Mat();

    // Masking
    Mat masked = new Mat();
    Mat maskedRGB = new Mat();

    // Contours
    List<MatOfPoint> contours = new ArrayList<>();
    Mat hierarchy = new Mat(); 

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
        
        output = input;

        // Find contours
        Imgproc.findContours(thresholded, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // Draw contours
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(output, contours, i, new Scalar(0, 255, 0));
        }

        // Get bounding box

        // return masked;
        // return thresholded;
        return output;
    }

}

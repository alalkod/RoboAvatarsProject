package org.firstinspires.ftc.teamcode;

import java.util.ArrayList;
import java.util.List;

import org.openftc.easyopencv.OpenCvPipeline;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Core;

import org.opencv.imgproc.Imgproc;

public class DetectionPipeline extends OpenCvPipeline {
    // Width, height of viewport
    public int width = 320;
    public int height = 240;

    Mat hsv = new Mat();

    // HSV thresholding
    //                                           degrees, percent, percent
    // image5
    public Scalar hsvLowerThreshold = new Scalar(100, 34, 0);
    public Scalar hsvUpperThreshold = new Scalar(280, 100, 100);
    // image6
    // public Scalar hsvLowerThreshold = new Scalar(215, 92, 37);
    // public Scalar hsvUpperThreshold = new Scalar(250, 100, 100);

    // Convert to 2 degrees, 0-255, 0-255
    Scalar hsvTrueLowerThreshold = new Scalar(hsvLowerThreshold.val[0] / 2, hsvLowerThreshold.val[1] * 255 / 100, hsvLowerThreshold.val[2] * 255 / 100);
    Scalar hsvTrueUpperThreshold = new Scalar(hsvUpperThreshold.val[0] / 2, hsvUpperThreshold.val[1] * 255 / 100, hsvUpperThreshold.val[2] * 255 / 100);
    Mat thresholded = new Mat();
    Mat thresholdedRGB = new Mat();

    // Masking
    Mat masked = new Mat();

    // Edge detection
    // image5
    public double cannyLowerThreshold = 120f;
    public double cannyUpperThreshold = 130f;
    // image 6
    // public double cannyLowerThreshold = 0f;
    // public double cannyUpperThreshold = 100f;

    Mat cannyEdge = new Mat();

    // Contours
    List<MatOfPoint> contours = new ArrayList<>();
    Mat hierarchy = new Mat(); 
    public int Ncontours;

    // Input with contours
    Mat imgContours = new Mat();

    // Bounding boxes
    public double bbWidthThreshold = 0.09 * width;
    public double bbHeightThreshold = 0.12 * height;
    List<Rect> bbs = new ArrayList<>();

    // Input with bounding boxes
    Mat imgBB = new Mat();

    // Output mat (with object detection)
    Mat output = new Mat();

    @Override
    public Mat processFrame(Mat input) {
        output = input;

        // Convert to hsv
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        
        // HSV threshold
        Core.inRange(hsv, hsvTrueLowerThreshold, hsvTrueUpperThreshold, thresholded);

        // Mask original pixel values over only the thresholded values (display/debug only)
        Imgproc.cvtColor(thresholded, thresholdedRGB, Imgproc.COLOR_GRAY2RGBA);
        Core.bitwise_and(input, thresholdedRGB, masked);

        // Canny edge detector
        Imgproc.Canny(masked, cannyEdge, cannyLowerThreshold, cannyUpperThreshold);

        // Find contours
        Imgproc.findContours(cannyEdge, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Ncontours = contours.size();

        imgContours = input.clone();

        // Draw contours
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(imgContours, contours, i, new Scalar(0, 255, 0));
        }

        imgBB = input.clone();

        // Find/draw bounding boxes, filter out noisy ones
        // Iterate through detected contours
        for (MatOfPoint contour : contours) {
            // Create a bounding box for it
            Rect rect = Imgproc.boundingRect(contour);
            // If rect is very small (very likely just noise), ignore
            // If it is above the set thresholds (percentage of the screen width/height), add it to the list of actual bbs
            // Also useful because we don't want to detect samples that are far away (small area on screen)
            if (rect.width >= bbWidthThreshold && rect.height >= bbHeightThreshold) {
                bbs.add(rect);
                Imgproc.rectangle(imgBB, rect, new Scalar(0, 0, 255));
            }
        }

        // return thresholded;
        // return masked;
        // return cannyEdge;
        // return imgContours;
        return imgBB;
        // return output;
    }

}

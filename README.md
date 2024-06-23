# shaper
 Create a target image from given shapes like triangles, cubes, circles, ... . There are two kinds of **images that work best**, **Highly Detailed Images:** These images have lots of fine details and intricate features. When blurred, the details still hold enough information to create a recognizable image and **Very Simple Images:** These images have clear, distinct shapes and minimal elements. Even when blurred, the simplicity makes them easy to recreate. The **worst image** you can give it is one where **color changes are not small enough to be ignored but still too complex**, making it difficult to use the basic shapes effectively.

<p float="center">
  <img src="/from_image.png" width="30%" />
  <img src="/middle_image.png" width="30%" />
  <img src="/final_image_ubuntu.png" width="30%" />
</p>

The aplustools version needed can be found [here](https://github.com/adalfarus/aplustools/tree/706087d7d69299766f4a9affbf1d5c4adb6d06ae).  I'll update it to the latest version when I have time to verify they work the same.

Arguments are the target image path, the shapes directory path, changing the color of the shapes, using random shapes, and finally if it should start at another image.
(This doesn't work sometimes)
````bash
python3 V4.py from_image.png shapes True False middle_image.png
````
You can leave out the last parameter if you want to start fresh like this:
````bash
python3 V4.py from_image.png shapes True False
````

When you close the window it'll save the image to final_image.png

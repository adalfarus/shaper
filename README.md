# shaper
 Create a target image from given shapes like triangles, cubes, circles, ...

Arguments are the target image path, the shapes directory path, changing the color of the shapes, using random shapes, and finally if it should start at another image.
````bash
py .\V4.py 003.JPG ./shapes True False .\final_image_old.png
````
You can leave out the last parameter if you want to start fresh like this:
````bash
py .\V4.py 8.jpg ./shapes True False
````
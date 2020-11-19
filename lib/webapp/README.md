### Back-End of Web Application

The `detector_app.py` script loads a pre-trained cuneiform sign detector and handles detection request from the [web front-end](https://github.com/to3i/cuneiform-sign-detection-webapp) of the web application.
It utilizes [Flask](https://palletsprojects.com/p/flask/) in order to make the detector available at a specific route (URL).

The webapp back-end receives a POST request from the front-end at http://localhost:5000/detector_php, runs the detector and returns a JSON response indicating if successful.
We recommend to run this back-end app locally and access it through the web front-end provided in the 
[cuneiform-sign-detection-webapp](https://github.com/to3i/cuneiform-sign-detection-webapp) repo.



#### Requirements

Additional Python packages:
- Flask (1.1.1) 
- Werkzeug (0.16.0)

#### Usage

- Config the `detector_app.py` as needed.
- Pre-trained model weights specified in the `detector_app.py` need to be placed in the `results/weights/` folder.
- Run `$ python detector_app.py` to start the back-end webapp and make it available to the front-end.



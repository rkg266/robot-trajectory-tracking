# robot-trajectory-tracking
The objective of this project is to achieve precise trajectory tracking for a ground differential-drive robot, aiming to address various real-life applications such as autonomous vehicles and other scenarios involving pathfinding and obstacle avoidance. The problem at hand can be formulated as a model-based infinite-horizon stochastic optimal control problem. To solve this problem, we will explore and compare two distinct approaches: (a) receding-horizon certainty equivalent control (Rh-CEC) and (b) generalized policy iteration (GPI). <br>

## Results:
Results for Rh-CEC are shown below: <br> <br>
![Rh-CEC error 460](/Rhcec_gifs/RhCEC_error_460.gif)

## Runnning code:
Files: utils.py, main3.py, GPI.py <br>
Dependencies: Python3, numpy, casadi, time  <br>

Running Rh-CEC:
1. Open main3.py
2. Edit the design parameters from line 196 onwards.
3. Run main3.py <br>
Output gifs: ./Rhcec_gifs


Running GPI:
1. Run GPI.py 

**For complete details about implementation, refer Project_report.pdf**

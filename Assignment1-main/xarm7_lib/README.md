# Install instructions
```
conda env create -f environment.yml
```

# Testing
```python
from xarm7_real import RealXArm7
robot = RealXArm7(ip='192.168.1.?')
robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
```

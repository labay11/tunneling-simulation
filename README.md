# Quantum tunneling simulation
Simulation of the tunneling effect of a gaussian wave packet in a infinite square box with a barrier at the middle.

You can use it using the command line as `python tunneling.py V0 L l T xi sigmax --TMAX --dt --filename` with

        V0                   height of the barrier in eV
        L                    half length of the box in nm
        l                    half length of the barrier in nm
        T                    kick in eV
        xi                   center of the gaussian
        sigmax               size of the gaussian

    optional arguments:
        -h, --help           show this help message and exit
        --TMAX TMAX          max time
        --dt dt              step in time
        --filename filename  animation destination file

Or you can import the script to your project and use it to create a new tunneling class

```python
tun = Tunneling(V0, L, l, T, xi, sx)
times = tun.experiment(TMAX, dt)
tun.plot(times, TMAX, dt, filename='somewhere/over/the/rainbow')
```

Example with a potential barrier of `5 eV` and a kick of intensity `4 eV`

![N|Solid](/examples/5_00_4_00.png)

# License
    Copyright 2018 labay11

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

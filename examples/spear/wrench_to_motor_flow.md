# Lee Controller to Motor Thrust Flow

```mermaid
flowchart LR
    A[Lee controller\nInput:\n- sDes = desired state\n  - desired position\n  - desired velocity\n  - desired acceleration\n  - desired yaw / yaw rate\n- current drone state\nOutput:\n- wrench_command = [Fx, Fy, Fz, Mx, My, Mz]]

    B[Wrench decomposition\nInput:\n- wrench_command\nOutput:\n- force_command = [Fx, Fy, Fz]\n- moment_command = [Mx, My, Mz]]

    C[Thrust sign conversion\nInput:\n- force_command[2] in NED or ENU\nOutput:\n- thrust_command = scalar total thrust]

    D[Control allocation / mixerFMinv\nInput:\n- t = [thrust_command, roll_torque, pitch_torque, yaw_torque]\n- mixerFMinv\nOutput:\n- w_squared_normalized = per-motor normalized squared speed]

    E[Scale to physical motor speed\nInput:\n- w_squared_normalized\n- w_max for each motor\nOutput:\n- w_squared_actual = per-motor speed^2\n- w_cmd = per-motor motor speed (rad/s)]

    F[DroneSimulator command inversion\nInput:\n- w_cmd\n- motor model parameters w_min, k\nOutput:\n- actions in [-1, 1]\n- fed to simulator]

    G[Motor dynamics / actual rotor speed\nInput:\n- actions\nOutput:\n- actual motor speed w_i]

    H[Propeller physics\nInput:\n- actual motor speed w_i\n- kTh, kTo\nOutput:\n- thrust_i = kTh * w_i^2\n- torque_i = kTo * w_i^2]

    A --> B --> C --> D --> E --> F --> G --> H
```

## Notes

- The controller produces a desired wrench, not thrust directly.
- The mixer/allocation matrix maps wrench to motor speed commands.
- The simulator then converts motor commands back into actual rotor speeds.
- Propeller constants define the motor-speed-to-thrust and motor-speed-to-torque relationship.
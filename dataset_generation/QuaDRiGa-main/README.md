# QuaDRiGa
QuaDRiGa, short for QUAsi Deterministic RadIo channel GenerAtor, is used for generating realistic radio channel impulse responses for system-level simulations of mobile radio networks. These simulations are used to determine the performance of new digital-radio technologies in order to provide an objective indicator for the standardization process in bodies like the third generation partnership program 3GPP.

QuaDRiGa was developed at Fraunhofer HHI to enable the modeling of MIMO radio channels for specific network configurations, such as indoor, satellite or heterogeneous configurations. Besides being a fully-fledged three dimensional geometry-based stochastic channel model, QuaDRiGa contains a collection of features created in SCM(e) and WINNER channel models along with novel modeling approaches which provide features to enable quasi-deterministic multi-link tracking of users (receiver) movements in changing environments.

QuaDRiGa contains a couple of new features and is furthermore calibrated against 3GPP channel models like 3GPP-3D and the latest New Radio channel model. The supported (standardized) channel models are:

* Compatibility with 3GPP TR 36.873 v12.5.0
* Compatibility with 3GPP TR 38.901 v16.1.0
* Compatibility with the mmMAGIC channel model (mmMAGIC D2.2)

Software License for The QuaDRiGa Channel Model  
© Copyright 2011 - 2021 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V., All rights reserved.

email: quadriga@hhi.fraunhofer.de

# Version history

## Added features in version 2.8:

* Site specific simulations
* Partial GPU acceleration using NVIDIA CUDA (optional)
* Octave 8 compatibility

## Added features in version 2.6:

* Octave 6.2 compatibility
* GPU acceleration in Octave (requires OCL package)
* Method to combine parameters of several different scenario configurations
* Method to add semi-deterministic clusters
* Method for calculating LSF and SSF parameters from MPCs
* Save and load-functions for channel data
* Performance improvements for larger scenarios

## Added features in version 2.4:

* Satellite channel modelling (multi-beam parabolic antennas, non-GSO satellite orbit model, TLE-data import, model parameters)
* 3GPP CDL and TDL models (TS 36.104 LTE, TR 38.901 NR, TR 37.885 V2X)
* 3GPP 38.901 InF models (absolute TOA model, InF parameters)
* Sement-by-segment channel generation

## Added features in version 2.2:

* Dual-mobility functionality
* Support for Industrial Indoor scenario
* Data Exchange Formats:
* QuaDRiGa Array Antenna Exchange Format (QDANT)
* QuaDRiGa Layout Exchange Format (based on KML)
* Updated technical documentation

## Features of version 2.0:

* Octave (v4.0) support
* 3D spatial consistency of large and small-scale fading based on the sum-of-sinusoids method
* Multi-frequency simulations (supporting carrier aggregation and functional split)
* Added support for scenarios: Indoor office, Rural Macrocell
* Supported frequency range: 500 MHz to 100 GHz carrier frequency, 100 MHz bandwidth  (2 GHz bandwidth for mmMAGIC models)
* Outdoor-to indoor penetration loss models (3GPP 38.901 and mmMAGIC models)
* Explicit ground reflection model

## Features of  version 1.4:

* Temporal evolution of the channel coefficients by updating the delays, the departure and arrival angles, the polarization, the shadow fading and the K-Factor
* Scenario transitions (including birth and death of scattering clusters)
* Variable speeds for mobile terminals
* Geometric polarization
* Spherical waves and support large array antennas
* Freely configurable array antenna support (including 3GPP antennas). Supports import of measured and simulated far-field antenna patterns.
* Supported scenarios: Urban Macrocell, Urban Microcell
* Supported frequency range: 500 MHz to 6 GHz carrier frequency, 100 MHz bandwidth

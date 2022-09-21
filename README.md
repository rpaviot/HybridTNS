# HybridTNS
 
 Code not ready just yet 
 
Code to compute redshift-space quantities (multipoles, wp(rp), Dsigma(rp)) with the Hybrid TNS model described in https://arxiv.org/abs/2007.08993
Example can be found in notebook.

Requirements : 
- numpy
- scipy 
- astropy 
- class : https://github.com/lesgourg/class_public
- respresso : http://www2.yukawa.kyoto-u.ac.jp/~takahiro.nishimichi/public_codes/respresso/index.html
- hankl for fftlog : https://arxiv.org/abs/2106.06331

To run the code :
python setup.py build_ext --inplace

One needs to modify the setup.py with its gcc environments.


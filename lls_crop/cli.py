"""lls_crop
Usage:
  lls_crop [-s SOURCE -c CHANNEL -x -m]
  lls_crop -h | --help  | --version

Options:
  -s <source>, --source <source>  Source directory [default:].
  -c <channel>, --channel <channel>  Channel for autocropping [default: 488].
  -x --crop  Flag if you want to crop [default: False].
  -m --copy  Flag if you want to copy to NVME [default: False].
  """
from docopt import docopt

__author__ = "Brandon Scott"
__version__ = "0.5.0"

import os

import llspy
from easygui import diropenbox
from multiprocess.pool import Pool

from lls_crop.base import (determine_filenames, read_crop_write,
                           read_crop_write_mip, robocopy)


def parse_args():
    """Use docopt to parse arguments if the command line is used."""
    args = docopt(__doc__, version=__version__,)
    source = args['--source']
    copy_switch = args['--copy']
    crop_switch = args['--crop']
    channel = args['--channel']
    return source, copy_switch, crop_switch, channel


def main() -> None:
    """1) Given a path from the microscope S:\\ transfer to the X:\\ drive
       2) Perform the deskew, deconvolution (or not) and rotation
       3) Crop down to the coverslip area
       4) Ensure that MIP maintains the proper metadata: Time, Pixel size, etc.
       5) Transfer the raw data and cropped data to the Z:\\ drive
       If crop_switch is True then the deconvolution will not be run. """
    source, copy_switch, crop_switch, channel = parse_args()

    if not source:
        source = diropenbox()
        if not source:
            print('Directory not chosen, exiting.')
            return
        source = source + "\\"
    os.chdir(source)

    print('Input Directory is: ', source)

    if copy_switch:
        destination = "X:\\DeconSandbox\\"
        if crop_switch:
            if not source.endswith("GPUdecon"):
                copy_source = source+"GPUdecon\\"
            else:
                copy_source = source
            copy_destination = "X:\\DeconSandbox\\GPUdecon\\"
        else:
            copy_source = source
            copy_destination = destination

        hidden_dir = "X:\\.empty\\"
        if not os.path.exists(hidden_dir):
            os.makedirs(hidden_dir)
        # Copy files using 64 threads, and no logging information displayed.
        robocopy(copy_source, copy_destination, '*.*')
    else:
        destination = source

    if not crop_switch:  # This will do the deskew, decon, and rotate
        lls_data = llspy.LLSdir(destination)
        options = {"correctFlash": False,
                   "nIters": 10,
                   "otfDir": 'Z:/data/LLSM_Alignment/OTF/',
                   "background": 95,
                   "bRotate": True,
                   "rMIP": (True, True, True),
                   "rotate": lls_data.parameters.angle}
        # Perform the deskewing, decon, and rotation.
        lls_data.autoprocess(**options)

    # Perform autocropping
    inputs = determine_filenames(
        source=destination+"GPUdecon\\", channel=channel)
    mip_path = destination+"GPUdecon\\MIPs\\"
    if os.path.exists(mip_path):
        mip_file = os.listdir(mip_path)[0]
        read_crop_write_mip(mip_path + mip_file, inputs[0][-1])

    Pool().map(read_crop_write, inputs)

    if copy_switch:
        # Move files using 64 threads, and no logging information displayed.
        os.chdir("X:\\")
        robocopy(destination, source, '*.txt', '/MOVE')
        robocopy(destination+"GPUdecon\\", source +
                 "GPUdecon\\", '*.*', '/MOVE', '/E')

        # Remove the raw data that was copied to destination.
        robocopy(hidden_dir, destination, '*.*', '/MIR')
        if os.path.exists(hidden_dir):
            os.removedirs(hidden_dir)


if __name__ == '__main__':
    main()

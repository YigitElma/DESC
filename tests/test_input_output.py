import unittest
import os
import pathlib
from desc.input_reader import InputReader
from desc.input_output import read_input


#class TestIO(unittest.TestCase):
#    """tests for input/output functions"""
#
#    def test_min_input(self):
#        dirname = os.path.dirname(__file__)
#        filename = os.path.join(dirname, 'MIN_INPUT')
#        inputs = read_input(filename)
#
#        self.assertEqual(len(inputs), 26)

class TestInputReader(unittest.TestCase):

    def test_no_input_file(self):
        self.argv0 = ['desc']
        self.assertRaises(NameError, InputReader(cl_args=self.argv0))

    def test_nonexistant_input_file(self):
        self.argv1 = self.argv0.append('nonexistant_input_file')
        self.assertRaises(FileNotFoundError, InputReader(cl_args=self.argv1))

    def test_min_input(self):
        self.argv2 = self.argv0.append('MIN_INPUT')
        ir = InputReader(cl_args=self.argv2)
        self.assertEqual(ir.args.prog, 'DESC', 'Program is incorrect.')
        self.assertEqual(ir.args.input_file, self.argv2[1],
                'Input file name does not match')
        self.assertEqual(ir.args.output_file, self.argv2[1] + '.output',
                'Default output file does not match.')
        self.assertEqual(ir.input_path,
                str(pathlib.Path('./'+self.argv2[1]).resolve()),
                'Path to input file is incorrect.')
        #Test defaults
        self.assertFalse(ir.args.plot, 'plot is not default False')
        self.assertFalse(ir.args.quiet, 'quiet is not default False')
        self.assertFalse(ir.args.verbose, 'verbose is not default False')
        #self.assertEqual(ir.args.vmec_path, '', "vmec path is not default ''")
        self.assertFalse(ir.args.gpuID, 'gpu argument was given')
        self.assertFalse(ir.args.numpy, 'numpy is not default False')
        self.assertEqual(os.environ['DESC_USE_NUMPY'], '', 'numpy environment '
            'variable incorrect with default argument')
        self.assertFalse(ir.args.version, 'version is not default False')
        self.assertEqual(len(ir.inputs), 26, 'number of inputs does not match '
            'number expected in MIN_INPUT')
        # test equality of arguments

    def test_np_environ(self):
        argv = self.argv2.append('--numpy')
        ir = InputReader(cl_args=argv)
        self.assertEqual(os.environ['DESC_USE_NUMPY'], 'True', 'numpy '
            'environment variable incorrect on use')

    def test_quiet_verbose(self):
        ir = InputReader(self.argv2)
        self.assertEqual(ir.inputs['verbose'], 1, "value of inputs['verbose'] "
            "incorrect on no arguments")
        argv = self.argv2.append('-v')
        ir = InputReader(argv)
        self.assertEqual(ir.inputs['verbose'], 2, "value of inputs['verbose'] "
            "incorrect on verbose argument")
        argv.append('-q')
        ir = InputReader(argv)
        self.assertEqual(ir.inputs['verbose'], 0, "value of inputs['verbose'] "
            "incorrect on quiet argument")

    def test_vmec_to_desc_input(self):
        pass


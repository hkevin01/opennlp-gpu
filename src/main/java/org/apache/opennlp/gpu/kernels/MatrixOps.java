package org.apache.opennlp.gpu.kernels;

import java.util.HashMap;
import java.util.Map;

import org.jocl.cl_kernel;
import org.jocl.cl_program;

/** Stub kernel manager */
public class MatrixOps {

  private Map<String,cl_program> programs = new HashMap<String,cl_program>();
  private Map<String,cl_kernel> kernels  = new HashMap<String,cl_kernel>();

  public cl_program getProgram(String name) { return programs.get(name); }
  public cl_kernel  getKernel(String name)  { return kernels.get(name);  }
}

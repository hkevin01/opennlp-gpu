package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.Map;

/** Stub default memory manager */
public class DefaultMemoryManager {

  private Map<Long,byte[]> pool = new HashMap<Long,byte[]>();

  public byte[] allocate(long size) {
    return new byte[(int)size];
  }

  public void free(byte[] data) {
    // no‐op
  }
}

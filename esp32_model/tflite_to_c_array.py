# Converts a .tflite file to a C header array for ESP32 TFLite Micro
# Usage: python tflite_to_c_array.py esp32_model/autoencoder_lstm_int8.tflite esp32_model/autoencoder_lstm_int8.h
import sys
import os

def tflite_to_c_array(tflite_path, header_path, var_name="autoencoder_lstm_int8_model"):
    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()
    array_str = ", ".join(str(b) for b in tflite_bytes)
    header = f"""
#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

const unsigned char {var_name}[] = {{
    {array_str}
}};
const unsigned int {var_name}_len = {len(tflite_bytes)};

#endif // {var_name.upper()}_H
"""
    with open(header_path, "w") as f:
        f.write(header)
    print(f"Saved: {header_path} ({len(tflite_bytes)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tflite_to_c_array.py <input.tflite> <output.h>")
        sys.exit(1)
    tflite_to_c_array(sys.argv[1], sys.argv[2])

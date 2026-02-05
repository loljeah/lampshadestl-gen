{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python interpreter
    python3

    # Python packages
    python3Packages.numpy
    python3Packages.numpy-stl
    python3Packages.matplotlib
    python3Packages.scipy
    python3Packages.tkinter

    # System dependencies for tkinter GUI
    tk
    tcl
  ];

  shellHook = ''
    echo "Lampshade STL Generator development environment"
    echo "Run: python lampshadegen-gui6.py"
  '';
}

with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";

  # Mandatory boilerplate for buildable env
  env = buildEnv { name = name; paths = buildInputs; };
  builder = builtins.toFile "builder.sh" ''
    source $stdenv/setup; ln -s $env $out
  '';

  # Customizable development requirements
  buildInputs = [
    # With Python configuration requiring a special wrapper
    (python37.buildEnv.override {
      ignoreCollisions = true;
      extraLibs = with python37Packages; [
        numpy
        tensorflow
      ];
    })
  ];

  shellHook = ''
    export SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt
  '';
}

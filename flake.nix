{
  description = "OpenPronounce – AI-powered English pronunciation feedback";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;       # needed for some CUDA bits if you enable them
            allowBroken = false;
          };
        };

        # ── Python interpreter + packages ──────────────────────────────────
        python = pkgs.python311;

        pythonEnv = python.withPackages (ps: with ps; [
          # core ML
          torch
          torchaudio
          transformers

          # audio processing
          librosa
          soundfile
          pydub

          # alignment / distance
          dtw-python
          fastdtw
          scipy
          scikit-learn

          # pronunciation
          phonemizer       # wraps espeak-ng; see shellHook for the path

          # NLP
          spacy

          # text-to-speech (lightweight gtts only; coqui-tts is not in nixpkgs –
          # install it via pip in a venv if you need TTS synthesis, see README below)
          gtts

          # Levenshtein
          python-levenshtein

          # web / API
          fastapi
          uvicorn
          python-multipart
          requests
          streamlit
          plotly

          # utilities
          numpy
          pip          # available so you can `pip install coqui-tts` in a venv
        ]);

      in {
        # ── Dev shell ──────────────────────────────────────────────────────
        devShells.default = pkgs.mkShell {
          name = "openpronounce";

          packages = [
            pythonEnv

            # system deps
            pkgs.ffmpeg          # audio decoding (packages.txt)
            pkgs.espeak-ng       # phonemizer backend
            pkgs.sox             # helpful for audio debugging
            pkgs.git
          ];

          shellHook = ''
            # Tell phonemizer where espeak-ng lives
            export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
            export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:$LD_LIBRARY_PATH"

            # Make sure the spacy model is downloaded on first use
            if ! python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
              echo "→ Downloading spaCy model en_core_web_sm …"
              python -m spacy download en_core_web_sm
            fi

            echo ""
            echo "  ╔══════════════════════════════════════════╗"
            echo "  ║        OpenPronounce dev shell           ║"
            echo "  ╠══════════════════════════════════════════╣"
            echo "  ║  FastAPI server:                         ║"
            echo "  ║    python -m uvicorn server:app \\        ║"
            echo "  ║      --host 0.0.0.0 --port 8000 --reload ║"
            echo "  ║                                          ║"
            echo "  ║  Streamlit app:                          ║"
            echo "  ║    streamlit run streamlit_app.py        ║"
            echo "  ║                                          ║"
            echo "  ║  CLI:                                    ║"
            echo "  ║    python cli.py file.wav \"Hello\"        ║"
            echo "  ╚══════════════════════════════════════════╝"
            echo ""
          '';
        };

        # ── Runnable apps ──────────────────────────────────────────────────
        apps = {
          # `nix run .#server`
          server = {
            type = "app";
            program = toString (pkgs.writeShellScript "openpronounce-server" ''
              export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
              export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:''${LD_LIBRARY_PATH:-}"
              exec ${pythonEnv}/bin/python -m uvicorn server:app \
                --host 0.0.0.0 --port 8000 --reload "$@"
            '');
          };

          # `nix run .#streamlit`
          streamlit = {
            type = "app";
            program = toString (pkgs.writeShellScript "openpronounce-streamlit" ''
              export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
              export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:''${LD_LIBRARY_PATH:-}"
              exec ${pythonEnv}/bin/streamlit run streamlit_app.py "$@"
            '');
          };

          # `nix run .` → default is the FastAPI server
          default = self.apps.${system}.server;
        };
      }
    );
}

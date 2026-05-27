{
  description = "OpenPronounce – AI-powered English pronunciation feedback";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      mkPythonEnv = pkgs:
        let python = pkgs.python312;
        in python.withPackages (ps: with ps; [
          # core ML
          torch
          torchaudio
          transformers

          # audio processing
          librosa
          soundfile
          pydub

          # alignment / distance
          (dtw-python.overrideAttrs (old: {
            postInstall = (old.postInstall or "") + ''
              rm -rf $out/lib/python*/site-packages/docs
            '';
          }))
          fastdtw
          scipy
          scikit-learn

          # pronunciation
          phonemizer

          # NLP
          spacy

          # text-to-speech
          gtts

          # Levenshtein
          levenshtein

          # web / API
          fastapi
          uvicorn
          python-multipart
          requests
          streamlit
          plotly

          # utilities
          numpy
          pip
        ]);
    in
    (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            allowBroken = false;
          };
        };

        pythonEnv = mkPythonEnv pkgs;
      in {
        # ── Buildable package ──────────────────────────────────────────────
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "openpronounce";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          dontBuild = true;

          installPhase = ''
            runHook preInstall

            mkdir -p $out/share/openpronounce
            cp -r . $out/share/openpronounce/

            mkdir -p $out/bin
            makeWrapper ${pythonEnv}/bin/python $out/bin/openpronounce-server \
              --add-flags "-m uvicorn server:app" \
              --set PHONEMIZER_ESPEAK_PATH "${pkgs.espeak-ng}/bin/espeak-ng" \
              --prefix LD_LIBRARY_PATH : "${pkgs.espeak-ng}/lib" \
              --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.espeak-ng pkgs.sox ]}" \
              --chdir "$out/share/openpronounce"

            makeWrapper ${pythonEnv}/bin/streamlit $out/bin/openpronounce-streamlit \
              --add-flags "run streamlit_app.py" \
              --set PHONEMIZER_ESPEAK_PATH "${pkgs.espeak-ng}/bin/espeak-ng" \
              --prefix LD_LIBRARY_PATH : "${pkgs.espeak-ng}/lib" \
              --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.espeak-ng pkgs.sox ]}" \
              --chdir "$out/share/openpronounce"

            runHook postInstall
          '';

          meta.mainProgram = "openpronounce-server";
        };

        # ── Dev shell ──────────────────────────────────────────────────────
        devShells.default = pkgs.mkShell {
          name = "openpronounce";

          packages = [
            pythonEnv
            pkgs.ffmpeg
            pkgs.espeak-ng
            pkgs.sox
            pkgs.git
          ];

          shellHook = ''
            export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
            export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:$LD_LIBRARY_PATH"

            export SPACY_MODELS_DIR="$HOME/.cache/openpronounce/spacy"
            mkdir -p "$SPACY_MODELS_DIR"
            export PYTHONPATH="$SPACY_MODELS_DIR:$PYTHONPATH"
            if ! python -c "import en_core_web_sm" 2>/dev/null; then
              echo "→ Downloading spaCy model en_core_web_sm …"
              pip install --target="$SPACY_MODELS_DIR" --quiet \
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
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
          server = {
            type = "app";
            program = toString (pkgs.writeShellScript "openpronounce-server" ''
              export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
              export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:''${LD_LIBRARY_PATH:-}"
              exec ${pythonEnv}/bin/python -m uvicorn server:app \
                --host 0.0.0.0 --port 8000 --reload "$@"
            '');
          };

          streamlit = {
            type = "app";
            program = toString (pkgs.writeShellScript "openpronounce-streamlit" ''
              export PHONEMIZER_ESPEAK_PATH="${pkgs.espeak-ng}/bin/espeak-ng"
              export LD_LIBRARY_PATH="${pkgs.espeak-ng}/lib:''${LD_LIBRARY_PATH:-}"
              exec ${pythonEnv}/bin/streamlit run streamlit_app.py "$@"
            '');
          };

          default = self.apps.${system}.server;
        };
      }
    )) // {
      # ── NixOS module (system-independent) ────────────────────────────────
      nixosModules.default = { config, lib, pkgs, ... }:
        let
          cfg = config.services.openpronounce;
        in {
          options.services.openpronounce = {
            enable = lib.mkEnableOption "OpenPronounce pronunciation feedback server";

            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
              description = "The OpenPronounce package to run.";
            };

            user = lib.mkOption {
              type = lib.types.str;
              default = "openpronounce";
            };

            group = lib.mkOption {
              type = lib.types.str;
              default = "openpronounce";
            };

            dataDir = lib.mkOption {
              type = lib.types.str;
              default = "/var/lib/openpronounce";
            };

            host = lib.mkOption {
              type = lib.types.str;
              default = "0.0.0.0";
            };

            port = lib.mkOption {
              type = lib.types.port;
              default = 8000;
            };

            environmentFile = lib.mkOption {
              type = lib.types.nullOr lib.types.path;
              default = null;
              example = "/run/secrets/openpronounce.env";
              description = "EnvironmentFile passed to the systemd unit.";
            };
          };

          config = lib.mkIf cfg.enable {
            users.users.${cfg.user} = {
              isSystemUser = true;
              group = cfg.group;
              home = cfg.dataDir;
              createHome = true;
            };
            users.groups.${cfg.group} = { };

            systemd.services.openpronounce = {
              description = "OpenPronounce pronunciation feedback server";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];

              serviceConfig = {
                User = cfg.user;
                Group = cfg.group;
                WorkingDirectory = cfg.dataDir;
                ExecStart = "${cfg.package}/bin/openpronounce-server --host ${cfg.host} --port ${toString cfg.port}";
                Restart = "on-failure";
                RestartSec = "5s";
                NoNewPrivileges = true;
                PrivateTmp = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                ReadWritePaths = [ cfg.dataDir ];
              } // lib.optionalAttrs (cfg.environmentFile != null) {
                EnvironmentFile = cfg.environmentFile;
              };
            };
          };
        };
    };
}

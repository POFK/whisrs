//! The `whisrsd` binary: argument handling and logging setup around
//! [`whisrs::daemon::run`], which carries the whole daemon.

use clap::Parser;

/// The daemon takes no options of its own, but declaring the interface means
/// `--version` and `--help` are answered and exit, instead of being ignored and
/// starting a daemon.
#[derive(Parser)]
#[command(name = "whisrsd", about = "whisrs dictation daemon", version)]
struct Args {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Not dead code: this call is what makes the flags above exit.
    Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    whisrs::daemon::run().await
}

#[cfg(test)]
mod tests {
    use clap::{error::ErrorKind, CommandFactory, Parser};

    use super::Args;

    /// Catches a malformed arg definition — duplicate names, conflicting
    /// shorts — the moment `Args` grows a real flag.
    #[test]
    fn daemon_args_are_valid() {
        Args::command().debug_assert();
    }

    /// `Args` exists only so `--version` and `--help` are answered and exit
    /// instead of being ignored and starting a daemon. Pin that behaviour,
    /// since nothing else in the daemon would notice if it regressed.
    ///
    /// `try_parse_from` rather than `parse`, which would exit the test harness.
    #[test]
    fn daemon_answers_version_and_help_instead_of_starting() {
        let Err(err) = Args::try_parse_from(["whisrsd", "--version"]) else {
            panic!("--version parsed as a normal start instead of exiting");
        };
        assert_eq!(err.kind(), ErrorKind::DisplayVersion);
        assert!(err.to_string().contains(env!("CARGO_PKG_VERSION")));

        let Err(err) = Args::try_parse_from(["whisrsd", "--help"]) else {
            panic!("--help parsed as a normal start instead of exiting");
        };
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
    }

    #[test]
    fn daemon_rejects_unknown_flags() {
        assert!(Args::try_parse_from(["whisrsd", "--nope"]).is_err());
    }

    /// `contrib/whisrs.service` runs `whisrsd` with no arguments, so a
    /// required field or positional on `Args` would break every install.
    #[test]
    fn daemon_parses_with_no_arguments() {
        assert!(Args::try_parse_from(["whisrsd"]).is_ok());
    }
}

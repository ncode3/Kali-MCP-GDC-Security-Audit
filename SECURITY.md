# Security Policy

## Supported Code

Security fixes target the current default branch. Draft, archived, and historical examples are not supported unless explicitly marked otherwise.

## Report a Vulnerability Privately

Do not open a public issue for a suspected vulnerability.

Email **nolan@atlanta-robotics.org** with the subject `Security report: Kali-MCP-GDC-Security-Audit`. Include:

- the affected file, component, and commit;
- impact and realistic attack conditions;
- safe reproduction steps;
- any recommended remediation;
- whether credentials or personal data may be involved.

Remove secrets and personal data from screenshots, logs, and sample payloads. Reports will be acknowledged as soon as practical.

## Testing Boundaries

Only test systems you own or have explicit written permission to assess. Do not disrupt services, access third-party data, use social engineering, or retain data obtained during testing.

## Secret Handling

Never commit credentials, private keys, tokens, production configuration, infrastructure state, or sensitive network details. Use environment variables or an approved secrets manager. If a secret is committed, revoke or rotate it immediately; deleting the current file does not remove it from Git history.

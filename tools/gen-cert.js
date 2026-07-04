// gen-cert.js — make a self-signed TLS cert (cert.pem + key.pem) so `deno task serve:https` can serve the demo
// over HTTPS on your LAN, which is what lets you open it on a phone.  (deno task cert)
//
// Why HTTPS: WebGPU only runs in a *secure context*. `localhost` qualifies, but a phone hitting your Mac's LAN
// IP over plain HTTP does NOT — so on the phone `navigator.gpu` is undefined and you get the red fallback. Any
// HTTPS origin is a secure context, even a self-signed one you tap past, so this is all it takes.
//
// The cert's SAN covers localhost, 127.0.0.1 and your current Wi-Fi IP so iOS offers the "visit website" tap
// (it rejects CN-only certs outright). It's still self-signed, so you'll see a one-time "not trusted" warning
// on the phone — tap through it. For a zero-warning setup instead, use mkcert (see the note at the bottom).
//
//   deno task cert              # generate cert.pem/key.pem (no-op if they already exist)
//   deno task cert -- --force   # regenerate, e.g. after your LAN IP changed
//
// Requires `openssl` on PATH (preinstalled on macOS). cert.pem/key.pem are gitignored — they never ship.

const force = Deno.args.includes('--force');

async function exists(path) {
  try {
    await Deno.stat(path);
    return true;
  } catch {
    return false;
  }
}

if (!force && (await exists('cert.pem')) && (await exists('key.pem'))) {
  console.log('cert.pem / key.pem already exist — run `deno task cert -- --force` to regenerate.');
  Deno.exit(0);
}

// Best-effort LAN IP (macOS): the active Wi-Fi/Ethernet interface. If none is found we fall back to a
// localhost-only cert (fine for the desktop, but a phone won't be able to validate the address).
async function lanIP() {
  for (const iface of ['en0', 'en1']) {
    try {
      const { success, stdout } = await new Deno.Command('ipconfig', {
        args: ['getifaddr', iface],
      }).output();
      if (success) {
        const ip = new TextDecoder().decode(stdout).trim();
        if (ip) return ip;
      }
    } catch {
      // ipconfig is macOS-only; elsewhere we just skip the IP SAN entry.
    }
  }
  return null;
}

const ip = await lanIP();
const san = ['DNS:localhost', 'IP:127.0.0.1', ip && `IP:${ip}`].filter(Boolean).join(',');

const { success } = await new Deno.Command('openssl', {
  args: [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
    '-keyout', 'key.pem', '-out', 'cert.pem',
    '-days', '825', '-subj', '/CN=windfoil-dev',
    '-addext', `subjectAltName=${san}`,
  ],
  stdout: 'inherit',
  stderr: 'inherit',
}).output();

if (!success) {
  console.error('openssl failed — is it installed and on PATH?');
  Deno.exit(1);
}

console.log(`\nWrote cert.pem + key.pem  (SAN: ${san})`);
console.log('Next:  deno task serve:https');
if (ip) console.log(`Then on your iPhone (same Wi-Fi) open:  https://${ip}:8080/demo/`);
console.log('\nZero-warning alternative: `brew install mkcert && mkcert -install`, then');
console.log(`  mkcert -cert-file cert.pem -key-file key.pem localhost 127.0.0.1${ip ? ` ${ip}` : ''}`);
console.log('and install mkcert\'s root CA on the phone (Settings → General → VPN & Device Management).');

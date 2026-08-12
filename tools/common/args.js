// args.js — the little `--flag value` / `--flag=value` reader the tools share.

/** Wrap an argv (e.g. Deno.args) in typed accessors. Exits with a message on a malformed number. */
export function args(argv) {
  const value = (name) => {
    const i = argv.indexOf(`--${name}`);
    if (i >= 0 && i + 1 < argv.length && !argv[i + 1].startsWith('--')) return argv[i + 1];
    const eq = argv.find((a) => a.startsWith(`--${name}=`));
    return eq ? eq.slice(name.length + 3) : null;
  };
  return {
    /** True if `--name` is present at all (with or without a value). */
    has: (name) => argv.some((a) => a === `--${name}` || a.startsWith(`--${name}=`)),
    /**
     * Bare arguments, skipping flags and the tokens they consume. A value-less flag directly before a bare
     * argument will swallow it (`--recursive dir` reads `dir` as --recursive's value), same ambiguity as
     * `string()` — so prefer the explicit `--name value` form when both are in play.
     */
    positionals: () => {
      const out = [];
      for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith('--')) {
          if (!argv[i].includes('=') && i + 1 < argv.length && !argv[i + 1].startsWith('--')) i++;
          continue;
        }
        out.push(argv[i]);
      }
      return out;
    },
    string: (name, fallback = null) => value(name) ?? fallback,
    number: (name, fallback) => {
      const raw = value(name);
      if (raw === null) return fallback;
      const v = Number(raw);
      if (!Number.isFinite(v)) {
        console.error(`--${name} must be a number, got "${raw}"`);
        Deno.exit(1);
      }
      return v;
    },
  };
}

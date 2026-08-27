#!/usr/bin/env python3
"""Mutation testing for the RTL.

A testbench that has only ever passed proves nothing: you have watched it
agree with a correct design, not disagree with a wrong one. This injects
one deliberate bug at a time and checks the suite notices.

    python tests/mutants.py            every module
    python tests/mutants.py requantize one module

A surviving mutant is one of two things. Either the testbench has a gap,
or the mutation is semantically identical to the original and no test
could tell them apart -- an equivalent mutant. The second kind is noted
inline where it is known.

Mutations are verified to apply before running. A mutation whose anchor
does not match would otherwise leave the file untouched, and the clean
design would pass, reporting a false survivor.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# (module, [(label, find, replace) or (label, find, replace, why_it_survives)])
#
# A fourth element marks a mutation known to be undetectable, with the
# reason. Those are reported separately rather than as failures: a
# mutation that cannot change behaviour at the tested parameters is not
# evidence of a weak testbench.
#
# Each mutation is chosen so a specific plausible mistake produces a
# different answer. A mutation the design could plausibly never make is
# not worth injecting.
MUTANTS = {
    'line_buffer': [
        ('window row and column swapped',
         'src_row = pos_row - PROW_W\'(PAD) + PROW_W\'(i)',
         'src_row = pos_row - PROW_W\'(PAD) + PROW_W\'(j)'),
        ('padding reads memory instead of zero',
         "in_range ? win[i][j][pos_ch] : '0",
         'win[i][j][pos_ch]'),
        ('window read at the input channel, not the output one',
         'win[i][j][pos_ch]',
         'win[i][j][channel]'),
        ('slot never wraps',
         "slot <= (slot == SLOT_W'(K - 2)) ? '0 : slot + 1'b1;",
         "slot <= slot + 1'b1;",
         'SLOT_W is 1 bit at K=3, so the counter wraps at 2 = K-1 by '
         'itself. K-1 is a power of two at K=3 and K=5; the explicit '
         'wrap first matters at K=7.'),
        ('drain removed',
         'wire                      advance  = accept || draining;',
         'wire                      advance  = accept;'),
        ('out_valid ignores the output geometry',
         'assign out_valid = pos_taken\n                    && (pos_row >= 0)',
         'assign out_valid = pos_taken; // (pos_row >= 0)',
         'pos_row and pos_col are counters that wrap at H_OUT and W_OUT, '
         'so they are never out of range and the test is always true. '
         'This is redundant logic in the design, not a gap in the tests.'),
    ],

    'mac_array': [
        ('signed dropped on the window operand',
         '$signed(`TAP(tap))', '`TAP(tap)'),
        ('signed dropped on the weight operand',
         '$signed(`WGT(pe, tap))', '`WGT(pe, tap)'),
        ('one tap dropped from the sum',
         'tap < K * K', 'tap < K * K - 1'),
        ('accumulator never reloads on the first channel',
         'if (first_channel)', 'if (1\'b0)'),
        ('bias ignored',
         '$signed(`BIAS(pe)) + $signed(prod[pe])', '$signed(prod[pe])'),
        ('out_valid on the first channel, not the last',
         'out_valid <= in_valid && last_channel;',
         'out_valid <= in_valid && first_channel;'),
    ],

    'requantize': [
        ('half-LSB dropped, so it truncates',
         "(prod + (P_W'(1) << (shift - 1))) >>> shift", 'prod >>> shift'),
        ('logical shift instead of arithmetic',
         '>>> shift\n                        : prod;', '>> shift\n                        : prod;'),
        ('low limit symmetric, losing a code',
         'OUT_LO = -(1 <<< (OUT_W-1))', 'OUT_LO = -((1 <<< (OUT_W-1)) - 1)'),
        ('saturation removed',
         "(rounded > OUT_HI) ? OUT_W'(OUT_HI) :\n            (rounded < OUT_LO) ? OUT_W'(OUT_LO) : OUT_W'(rounded)",
         "OUT_W'(rounded)"),
        ('mult_neg ignored, so no leaky slope',
         '$signed(`ACC(pe)) * mult_neg', '$signed(`ACC(pe)) * mult'),
        ('out_valid sets and never clears',
         'else                out_valid <= in_valid;',
         "else if (in_valid) out_valid <= 1'b1;"),
    ],
}


def _wrap(text, width=62):
    out, line = [], ''
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f'{line} {word}'.strip()
    if line:
        out.append(line)
    return out


def run(module, mutations):
    rtl = ROOT / 'rtl' / f'{module}.sv'
    tb = ROOT / 'tb' / f'tb_{module}.sv'
    if not tb.exists() or not rtl.exists():
        print(f'  {module}: no testbench or module, skipped')
        return 0, 0

    clean = rtl.read_text()
    killed = survived = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dut, exe = tmp / f'{module}.sv', tmp / 'a.out'

        def simulate():
            c = subprocess.run(['iverilog', '-g2012', '-o', str(exe), str(tb), str(dut)],
                               capture_output=True, text=True)
            if c.returncode:
                return 'DID NOT COMPILE'
            out = subprocess.run(['vvp', str(exe)], capture_output=True, text=True).stdout
            for line in out.splitlines():
                if line.startswith(('PASS', 'FAIL')):
                    return line.split()[0]
            return 'NO VERDICT'

        # The control has to pass, or every "killed" below is meaningless.
        dut.write_text(clean)
        if simulate() != 'PASS':
            print(f'  {module}: CONTROL FAILS -- fix the design before mutating')
            return 0, 0

        for entry in mutations:
            label, find, replace = entry[:3]
            expected = entry[3] if len(entry) > 3 else None
            # Verify the mutation applies. An anchor that does not match
            # would leave the design clean, and the pass would be
            # reported as a survivor.
            if find not in clean:
                print(f'    {label:<48} ANCHOR STALE')
                continue
            dut.write_text(clean.replace(find, replace, 1))
            verdict = simulate()
            if verdict == 'FAIL':
                killed += 1
                print(f'    {label:<48} killed')
            elif expected:
                print(f'    {label:<48} survives, as expected')
                for line in _wrap(expected):
                    print(f'      {line}')
            else:
                survived += 1
                print(f'    {label:<48} SURVIVED ({verdict})')

    return killed, survived


def main():
    targets = sys.argv[1:] or sorted(MUTANTS)
    total_k = total_s = 0
    for module in targets:
        if module not in MUTANTS:
            print(f'  no mutations defined for {module}')
            continue
        print(f'  {module}')
        k, s = run(module, MUTANTS[module])
        total_k += k
        total_s += s
        print()

    print(f'  {total_k} killed, {total_s} survived')
    if total_s:
        print('  a survivor is either a gap in the testbench or a mutation')
        print('  that cannot change behaviour. Work out which.')
    return 1 if total_s else 0


if __name__ == '__main__':
    sys.exit(main())

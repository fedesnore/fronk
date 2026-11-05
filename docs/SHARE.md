# Share your cluster

A report from your own machines helps other people judge whether their hardware
will work and helps improve Fronk. Working setups, blocked installations and
small fixes are all worth sharing. You do not need a new backend or a benchmark
suite to contribute.

## If installation is blocked

On GitHub, open **Issues → New issue** and choose **Installation help**.
If an issue already describes the same problem, add your observations there.
Start with just:

- The computer's role and operating system.
- The step that stopped and its exact error.
- The last step that worked, if you know it.
- The Fronk release or download you used.

For example: “Node 2, Debian Linux. The manager built the program, but Verify
on the head does not find it. The node says its service started. Both computers
are connected to the same Ethernet switch.” Add a short log excerpt when one
is available. You can ask for help before you know the cause.

[Troubleshooting](TROUBLESHOOTING.md) and [the debugging brief](llm/DEBUG.md)
can help you make progress while preparing the issue. An assistant should ask
for further details only as the investigation needs them.

## If the cluster works

In the head console, scroll to **Report → Save cluster report**. It downloads
a Markdown file containing machine hardware, assigned units and the existing
test's rate and timing. Nothing is uploaded. Hostnames, addresses, pairing
codes, paths and chat text are omitted. Fill in the physical cables, dense GPU
engine and whether the run was cold or warm, review the file, then paste it
into **Issues → New issue → Cluster result**. You can attach a screenshot too.

From a terminal on the head, the same export is available with:

```sh
python3 ~/fronk/src/tools/fronk_report.py --output cluster-result.md
```

Keep the head console open for that command. It does not run another benchmark
or overwrite an existing report. The [example cluster](TEST_CLUSTER.md) shows
the kind of result that is useful. A few lines about a failure are enough too.

Open **Issues → New issue** and choose **Cluster result**. Include a small
machine table:

| Label / role | CPU and GPU | RAM | OS | Fronk engine or backend | Units in RAM / on SSD |
|---|---|---|---|---|---|
| Head | | | | Dense engine; local expert backend if used | |
| Node 1 | | | | | |
| Node 2 | | | | | |

Use labels such as `head` and `node-1`; personal hostnames are unnecessary.
On the head, distinguish the dense GPU engine from its local expert backend.
Copy the actual unit counts and backend names from the console's plan.

Describe the cables too: direct Thunderbolt ports, or the speed of the
head-to-switch link and each node link. Note a shared USB hub or uplink if one
is involved. A short sketch is enough:

```text
head -- [link type/speed] -- switch -- [link type/speed] -- node-1
                                 `-- [link type/speed] -- node-2
```

Then say what you tried and what happened. A completed demo and a real chat are
different results. If reporting speed, include the prompt (or a harmless
reproduction), context/output length if known, whether the run was cold or warm,
and where the number came from. Label a console rate, an API token count and a
demo estimate as such. “I don't know” is better than inventing a measurement.

No extra calibration or model download is needed just to share a result.
Existing console output is a useful starting point. For a repeatable performance
comparison, the [measurement guide](MEASUREMENTS.md) explains the time intervals.

## If you found a solution

Return to the issue and record:

1. What failed and on which machine.
2. What you changed: setting, command, cable, file or code.
3. What passed afterward: the same failed step, the demo, or a completed chat.

Keep an uncertain cause marked as uncertain. A workaround can still be useful.
If the solution belongs in Fronk, send a focused pull request linked to the
issue. It may be a clearer instruction, a better error message, a regression
test or a code fix. A new backend should include its build command, hardware
requirements and numerical test results. [CONTRIBUTING.md](../CONTRIBUTING.md)
describes the development workflow.

## Get help preparing the report

Give a frontier model [REPORT.md](llm/REPORT.md), then ask:

> Help me prepare a Fronk issue from this setup and these observations. Separate
> what worked, what failed and what I only suspect. Use neutral machine labels,
> remove secrets and show me the draft before anything is posted.

Review the draft yourself. Never post the pairing code, `cluster.secret`, SSH
keys, the model file, private prompts or an unreviewed state directory. A small
relevant excerpt is usually enough. Keep the issue updated when you find a
solution so someone with the same hardware can follow it.

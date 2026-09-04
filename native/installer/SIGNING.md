# Code signing

The installer and the app exe are **not signed**. This is the single biggest friction
point in distribution: Windows SmartScreen shows a full-screen "Windows protected your
PC" warning on an unsigned installer from an unknown publisher, and the user has to
click through *More info → Run anyway* to proceed. The `/draft-coach` page already
warns about this, but a warning is not a fix — most people stop there.

Signing can't be done from this repo alone: it needs a certificate that must be bought
and kept secret, so it is a decision and a purchase, not a code change.

## What to buy

An **OV (Organisation Validation) code-signing certificate** is the cheap option (~$200-400/yr)
but note that since 2023 all code-signing certs must be issued on hardware (a USB token
or a cloud HSM), so there is no plain `.pfx` file to drop in a build. OV also builds
SmartScreen reputation slowly — the warning may persist for a while after signing.

An **EV (Extended Validation) certificate** (~$400-700/yr) gets SmartScreen reputation
immediately, which is the actual thing being bought here.

Either way you need a registered organisation to be validated against; individuals can
generally only get OV.

## Wiring it up once you have one

Inno Setup signs via a named "sign tool" configured in the IDE or passed to ISCC:

```
ISCC.exe /Ssigntool="$q<path to signtool.exe>$q sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 $f" ^
         /O"C:\Users\Umake\Documents\HotSFever-dist" native\installer\HotSFeverDraftCoach.iss
```

Then add to `[Setup]` in `HotSFeverDraftCoach.iss`:

```
SignTool=signtool
SignedUninstaller=yes
```

Sign the app exe too, not just the installer — SmartScreen checks the thing that runs,
and an unsigned exe inside a signed installer still warns on launch.

**Always timestamp** (`/tr`). Without it every signature expires with the certificate,
and previously-shipped builds start warning again.

## Until then

The portable zip is the workaround worth pointing people at: extracting and running an
exe raises a milder prompt than an unsigned installer does. Both are offered on
`/draft-coach`.

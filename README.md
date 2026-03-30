# NeuralSync Core

> Distributed compute engine for federated gradient computation

## Build Dependencies

This repo makes use of git submodules.

The first time you compile, or perhaps after a big update after a `git pull`, you need to update the submodules:

```bash
git submodule init
git submodule update
```

### Mac

Install [XCode](https://apps.apple.com/za/app/xcode/id497799835?mt=12) and then the XCode Command Line Tools with the following command

```
xcode-select --install
```

For macOS Mojave additional headers need to be installed, run

```
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

and follow the prompts

Install Brew

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Run the following to install needed bottles

```
brew install git
brew install cmake
```

### Linux

Run the following to install dependencies

```
apt-get install git cmake libc++-dev libc++abi-dev
```

### Windows

Install [Git](https://git-scm.com/download/win)

Install [CMake](https://cmake.org/download/)

Install [Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)

## Build

```bash
cargo build --release
```

## Usage

Start the coordinator:
```bash
./target/release/coordinator
```

Start the engine:
```bash
./target/release/engine
```

Or use the Python interface:
```bash
maturin develop --features python --release
python app.py
```

# Troubleshooting

## Mac/OSX

If you're experiencing linker issues, or messages like

`cstdint:153:8: error: no member named 'int8_t' in the global namespace`

then you might have multiple conflicting versions of clang installed.

Try:

- Does `which cc` report more than one binary? If so, uninstalling one of the clang compilers might help.
- Upgrading cmake. `brew uninstall cmake && brew install cmake`
- `cargo clean`

On Apple ARM64 hardware and newer XCode releases, the compute backend might fail some tests.
Building using an older SDK might help. Find location of current SDKs with `xcrun --show-sdk-path`, then for example:
```bash
export NSYNC_CMAKE_OSX_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX12.3.sdk"
cargo build
```

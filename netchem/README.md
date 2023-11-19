# NetChem

---
- [NetSci](../README.md)
- [CuArray](../cuarray/README.md)
# Overview

---

# API Documentation

  <details><summary><b>Classes</b></summary>

- [Atom](#atom-class)
- [Atoms](#atom-class)
- [Node](#atom-class)
- [Network](#atom-class)

</details>

---

## Atom ___class___

- **Language**: C++
- **Description**: Represents an atom in a molecular network.
- <details><summary><b>Methods</b></summary>

  <details><summary><b>C++</b></summary>

    - [`Atom()`](#atom)
    - [`Atom(const std::string &pdbLine)`](#atomconst-stdstring-pdbline)
    - [`Atom(const std::string &pdbLine, int atomIndex)`](#atomconst-stdstring-pdbline-int-atomindex)
    - [`int index() const`](#int-index-const)
    - [`std::string name()`](#stdstring-name)
    - [`std::string element()`](#stdstring-element)
    - [`std::string residueName()`](#stdstring-residuename)
    - [`int residueId() const`](#int-residueid-const)
    - [`std::string chainId()`](#stdstring-chainid)
    - [`std::string segmentId()`](#stdstring-segmentid)
    - [`float temperatureFactor() const`](#float-temperaturefactor-const)
    - [`float occupancy() const`](#float-occupancy-const)
    - [`int serial() const`](#int-serial-const)
    - [`std::string tag()`](#stdstring-tag)
    - [`float mass() const`](#float-mass-const)
    - [`unsigned int hash() const`](#unsigned-int-hash-const)
    - [`float x(CuArray<float> *coordinates, int frame, int numFrames) const`](#float-xcuarrayfloat-coordinates-int-frame-int-numframes-const)
    - [`float y(CuArray<float> *coordinates, int frame, int numFrames) const`](#float-ycuarrayfloat-coordinates-int-frame-int-numframes-const)
    - [`float z(CuArray<float> *coordinates, int frame, int numFrames) const`](#float-zcuarrayfloat-coordinates-int-frame-int-numframes-const)
    - [`void load(const std::string &jsonFile)`](#void-loadconst-stdstring-jsonfile)

  </details>

  <details><summary><b>Python</b></summary>

  </details>

</details>





---

### C++ Methods

---

#### `Atom()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Default constructor for Atom. Constructs an empty Atom object.

---

#### `Atom(const std::string &pdbLine)`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Constructor for Atom with PDB line. Constructs an Atom object using the provided PDB line, parsing it to extract relevant atom information.
- **Parameters**:
  - `const std::string &pdbLine`: The PDB line containing atom information in the standard PDB format.

---

#### `Atom(const std::string &pdbLine, int atomIndex)`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Constructor for Atom with PDB line and atom index. Constructs an Atom object using the provided PDB line and atom index.
- **Parameters**:
  - `const std::string &pdbLine`: The PDB line containing atom information.
  - `int atomIndex`: The atom index.

---

#### `int index() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the atom index.
- **Returns**: The atom index.

---

#### `std::string name()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the atom name.
- **Returns**: The atom name.

---

#### `std::string element()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the atom element.
- **Returns**: The atom element.

---

#### `std::string residueName()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the residue name.
- **Returns**: The residue name.

---

#### `int residueId() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the residue ID.
- **Returns**: The residue ID.

---

#### `std::string chainId()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the chain ID.
- **Returns**: The chain ID.

---

#### `std::string segmentId()`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the segment ID.
- **Returns**: The segment ID.

---

#### `float temperatureFactor() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the temperature factor.
- **Returns**: The temperature factor.

---

#### `float occupancy() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the occupancy.
- **Returns**: The occupancy.

---

#### `int serial() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the serial number, which is one greater than the atom index.
- **Returns**: The serial number.

---

#### `std::string tag()`
- **Language**: C++
- **Class**: Atom
- **Description**: Get the atom tag, which is the concatenation of the residue name, residue ID, chain ID, and segment ID.
- **Returns**: The atom tag.

---

#### `float mass() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the mass of the atom.
- **Returns**: The mass of the atom.

---

#### `unsigned int hash() const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the hash of the atom, calculated from the atom tag and index.
- **Returns**: The hash of the atom.

---

#### `float x(CuArray<float> *coordinates, int frame, int numFrames) const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the x-coordinate of the atom.
- **Parameters**:
  - `CuArray<float> *coordinates`: The CuArray containing the coordinates.
  - `int frame`: The frame index.
  - `int numFrames`: The total number of frames.
- **Returns**: The x-coordinate of the atom.

---

#### `float y(CuArray<float> *coordinates, int frame, int numFrames) const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the y-coordinate of the atom.
- **Parameters**:
  - `CuArray<float> *coordinates`: The CuArray containing the coordinates.
  - `int frame`: The frame index.
  - `int numFrames`: The total number of frames.
- **Returns**: The y-coordinate of the atom.

---

#### `float z(CuArray<float> *coordinates, int frame, int numFrames) const`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Get the z-coordinate of the atom.
- **Parameters**:
  - `CuArray<float> *coordinates`: The CuArray containing the coordinates.
  - `int frame`: The frame index.
  - `int numFrames`: The total number of frames.
- **Returns**: The z-coordinate of the atom.

---

#### `void load(const std::string &jsonFile)`
- **Language**: C++
- **Class**: [Atom](#atom-class)
- **Description**: Load atom information from a JSON file.
- **Parameters**:
  - `const std::string &jsonFile`: The name of the JSON file to load.






 
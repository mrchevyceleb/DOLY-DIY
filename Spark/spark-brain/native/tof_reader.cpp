// Share the SDK's existing ToF instance; never initialize hardware here.
// The stock Python getter holds the GIL while waiting for native updates.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <TofControl.h>
#include <exception>
#include <string>
#include <vector>

static PyObject* read_sensors(PyObject*, PyObject*) {
    std::vector<TofData> samples;
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try { samples = TofControl::getSensorsData(); }
    catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "Native ToF read failed"; }
    Py_END_ALLOW_THREADS
    if (!error.empty()) {
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    PyObject* result = PyList_New(samples.size());
    if (!result) return nullptr;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        PyObject* row = Py_BuildValue("iiiL", static_cast<int>(s.side), s.range_mm,
                                     static_cast<int>(s.error), static_cast<long long>(s.update_ms));
        if (!row) { Py_DECREF(result); return nullptr; }
        PyList_SET_ITEM(result, i, row);
    }
    return result;
}
static PyMethodDef methods[] = {
    {"read_sensors", read_sensors, METH_NOARGS, "Read existing SDK sensors without holding the GIL."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "spark_tof_native", nullptr, -1, methods};
PyMODINIT_FUNC PyInit_spark_tof_native() { return PyModule_Create(&module); }

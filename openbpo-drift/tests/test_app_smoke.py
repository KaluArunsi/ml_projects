def test_streamlit_app_imports():
    import app

    assert app.PROJECT_ROOT.name == "openbpo-drift"

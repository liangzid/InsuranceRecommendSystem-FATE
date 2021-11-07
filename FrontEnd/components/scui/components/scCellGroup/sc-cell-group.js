"use strict";

var checkboxGroupCount = 1;
Component({
  properties: {
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    value: []
  },
  externalClasses: ["sc-class"],
  ready: function () {
    this.items = this._getAllCheckboxs();
  },
  methods: {
    _getAllCheckboxs: function () {
      return getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scCell-' + this.data.swanIdForSystem);
    }
  }
});
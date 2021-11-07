
var config = require('../../config');
var app = getApp();
Page({
  data: {
    index: 1,
    childNum: 0,
    hasChild: false,
    houseNum: 0,
    hasHouse: false,
    isFemale:false,
    questionnaire: {
      formPicker1: '我该称呼您是先生还是女士?:',
      formPicker2: '您的年龄是:',
      formPicker3: '您有下列哪些生活习惯:',
      formPicker3_1: '离婚后与何方共同生活?',
      formPicker3_2: '另一方的探望方式？',
      formPicker4: '您有下列哪些工作习惯:',
      formPicker4_1: '房产是否有房贷?',
      formPicker5: '您的常用出行方式有哪些:',
      formPicker6: '您的年收入大概是:',
      formPicker7: '您有无社保:'
      //formPicker8: '您:'
    },
  },
  indexPlus: function () {
    var that = this;
    var tempindex = this.data.index + 1;
    this.setData({
      index: tempindex
    });
  },
  indexMinus: function () {
    var that = this;
    var tempindex = this.data.index - 1;
    this.setData({
      index: tempindex
    });
  },
  resetUpload: function(e){
      console.log("数据清理,重新提交。")
      swan.navigateTo({
          url: '../upload/upload'
      });
  },
  judgeFemale: function(e){
      console.log(e)
      if(e.detail.value == '1'){
        this.setData({
            isFemale:true,
        })
        // indexPlus: function () {
        //     var that = this;
        //     var tempindex = this.data.index + 1;
        //     this.setData({
        //       index: tempindex
        //     });
        //   }
      }
      else{
        this.setData({
            isFemale:false,
        })
      }

  },
  formSubmit: function (e) {
    console.log(e);
    console.log('in form submit')
    var formdata = e.detail.value;
    console.log('form发生了submit事件，携带数据为：', e.detail.value);
    swan.navigateTo({
        url: '../success2/success2'
      });
    swan.request({
      url: '',
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      method: "POST",
      data: formdata,
      success: function (res) {
        console.log(res);
        console.log(res.data);
        if (res.statusCode == 200) {
          swan.showToast({
            title: '提交成功',
            icon: 'success'
          });
          swan.navigateTo({
            url: '../success/success?gendoc=' + res.data
          });
        }
        else {
          swan.showToast({
            title: '提交失败',
            icon: 'loading'
          });
          swan.navigateTo({
            url: '../success2/success2'
          });
        }
      }
    });
  },

  bindKeyChildInput(e) {
    console.log(e.detail.value)
    if(e.detail.value != ''){
        this.setData({
            childNum : e.detail.value,
            hasChild : true
        });
    }
    else{
        this.setData({
            childNum : 0,
            hasChild : false
        });
    }
    this.data.childNum = parseInt(this.data.childNum)
  },

  bindKeyHouseInput(e) {
    console.log(e.detail.value)
    if(e.detail.value != ''){
        this.setData({
            houseNum : e.detail.value,
            hasHouse : true
        });
    }
    else{
        this.setData({
            houseNum : 0,
            hasHouse : false
        });
    }
    this.data.houseNum = parseInt(this.data.houseNum)
  },

  onReady: function () {},
  onShow: function () {},
  onHide: function () {},
  onUnload: function () {},

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {},

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {},

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {
    return {
      title: '法律协议一键生成',
      desc: '法律协议一键生成',
      path: '/pages/index/index'
    };
  }
});
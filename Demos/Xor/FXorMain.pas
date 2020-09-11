unit FXorMain;

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, daNeuralNet, daNeuralNet.Dense,
  Vcl.StdCtrls, Vcl.Buttons, Vcl.ExtCtrls;

type
  TForm1 = class(TForm)
    BBOneEpoch: TBitBtn;
    BBOneHundredEpochs: TBitBtn;
    Image1: TImage;
    Label1: TLabel;
    BBReset: TBitBtn;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure BBOneEpochClick(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure BBOneHundredEpochsClick(Sender: TObject);
    procedure BBResetClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    FNetwork : TdaNeuralNet;
    FRender : TBitmap;
    FTrainInputs : TdaNNDataset;
    FTrainOutputs : TdaNNDataset;

    procedure Render;
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

procedure TForm1.FormCreate(Sender: TObject);
begin
   RandSeed := 0;

   // initialize network with 2 inputs, one hidden layer of size 2 and an output layer of size 1
   FNetwork := TdaNeuralNet.Create;
   FNetwork.AddLayer(TdaNNInputLayer.Create(2));
   FNetwork.AddLayer(TdaNNDenseLayer.Create(2, daNNActivation.Sigmoid, 0.3, 0.9));
   FNetwork.AddLayer(TdaNNDenseLayer.Create(1, daNNActivation.Sigmoid, 0.3, 0.9));
   FNetwork.Build([ nnboForTraining ]);
   FNetwork.RandomizeWeights;

   // training set is truth table for XOR
   FTrainInputs  := [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ];
   FTrainOutputs := [ [   0  ], [   1  ], [   1  ], [   0  ] ];

   FRender := TBitmap.Create;
   FRender.PixelFormat := pf32bit;
   FRender.SetSize(64, 64);
end;

procedure TForm1.FormDestroy(Sender: TObject);
begin
   FRender.Free;
   FNetwork.Free;
end;

procedure TForm1.FormShow(Sender: TObject);
begin
   if Image1.Picture.Width = 0 then
      Render;
end;

procedure TForm1.BBOneEpochClick(Sender: TObject);
begin
   FNetwork.TrainSet(FTrainInputs, FTrainOutputs);
   Tag := Tag + 1;
   Render;
end;

procedure TForm1.BBOneHundredEpochsClick(Sender: TObject);
begin
   for var i := 1 to 100 do
      FNetwork.TrainSet(FTrainInputs, FTrainOutputs);
   Tag := Tag + 100;
   Render;
end;

procedure TForm1.BBResetClick(Sender: TObject);
begin
   FNetwork.RandomizeWeights;
   Tag := 0;
   Render;
end;

// Render
//
procedure TForm1.Render;
begin
   var inputs : TdaNNSingleArray;
   SetLength(inputs, 2);
   for var y := 0 to FRender.Height-1 do begin
      inputs[1] := y / FRender.Height;
      var p := PCardinal(FRender.ScanLine[y]);
      for var x := 0 to FRender.Width-1 do begin
         inputs[0] := x / FRender.Width;
         var output := 2*(FNetwork.Run(inputs)[0] - 0.5);

         // turn output into bright colors
         if output < 0 then
            output := 3*(0.5-0.5*Sqr(output))
         else output := 3*(0.5+0.5*Sqr(output));
         if output < 1 then
            p^ := Round(output*255)*$10000
         else if output < 2 then
            p^ := Round((2 - output)*255)*$10000 + Round((output - 1)*255)*$100
         else p ^ := Round((3 - output)*255)*$100 + Round((output - 2)*255);
         Inc(p);
      end;
   end;
   Image1.Picture.Graphic := FRender;

   Caption := 'Epoch = ' + IntToStr(Tag);

   Label1.Caption := Caption + #10#10;
   for var a := 0 to 1 do for var b := 0 to 1 do begin
      inputs[0] := a;
      inputs[1] := b;
      Label1.Caption := Label1.Caption
                      + Format('%d %d -> %.05f', [ a, b, FNetwork.Run(inputs)[0] ]) + #13#10;
   end;

end;

end.

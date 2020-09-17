program MNIST_digits;

uses
  Vcl.Forms,
  FMNIST in 'FMNIST.pas' {FormMNIST},
  daNeuralNet.JIT in '..\..\Source\daNeuralNet.JIT.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TFormMNIST, FormMNIST);
  Application.Run;
end.

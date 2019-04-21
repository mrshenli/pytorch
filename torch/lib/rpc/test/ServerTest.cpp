#include <rpc/Server.hpp>

#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>
#include <ATen/ATen.h>
#include <torch/types.h>



using Symbol = torch::jit::Symbol;
using IValue = torch::jit::IValue;

int main(int argc, char** argv) {

  std::string name = "aten::add";

  Symbol symbol = Symbol::fromQualString("aten::add");
  std::cout << "=== got symbol == " << symbol << std::endl << std::flush;
  std::vector<IValue> args = {};
  args.push_back(IValue(at::ones({2, 2})));
  args.push_back(IValue(at::ones({2, 2})));
  args.push_back(IValue(1));

  rpc::Request request(symbol, args);

  std::cout << "=== created request \n" << std::flush;

  rpc::Server server;

  auto response = server.processRequest(request);

  std::cout << "=== done processing request \n" << std::flush;


  for (auto value: response.values_) {
    std::cout << " === return value is " << value << std::flush << std::endl;
  }


  args.clear();
  args.push_back(IValue(at::ones({2, 2})));
  args.push_back(IValue(at::ones({2, 2})));
  args.push_back(IValue(1));

  std::vector<at::Tensor> pickler_tensor_table;

  torch::jit::Pickler pickler(&pickler_tensor_table);
  pickler.start();

  for (auto arg: args) {
    pickler.addIValue(arg);
  }

  pickler.addIValue(IValue(name));

  pickler.finish();

  auto tensor = torch::from_blob((void *)pickler.stack().data(), pickler.stack().size(), {torch::kChar});



  std::cout << "=== start unpickling " << pickler.stack().size() << " bytes " << std::flush;
  std::vector<at::Tensor> unpickler_tensor_table;
  //torch::jit::Unpickler unpickler(tensor.data_ptr(), tensor.numel(), &unpickler_tensor_table);
  char buffer[1024];
  std::memcpy(buffer, pickler.stack().data(), pickler.stack().size());
  torch::jit::Unpickler unpickler(buffer, pickler.stack().size(), &unpickler_tensor_table);
  // checkout ScriptModuleSerializer::writeTensorTable
  std::cout << "=== done creating unpickler, retrieving " << tensor.numel() << " bytes" << std::flush;

  std::vector<IValue> parsed_values = unpickler.parse_ivalue_list();
  std::cout << "=== done unpickling \n" << std::flush;


  torch::jit::IValue parsed_name = parsed_values.back();
  parsed_values.pop_back();

  AT_CHECK(parsed_name.isString(), "first ivalue must be string");

  Symbol parsed_symbol = Symbol::fromQualString(parsed_name.toStringRef());
  rpc::Request parsed_request(parsed_symbol, parsed_values);

  std::cout << "=== done creating request \n" << std::flush;


  auto parsed_response = server.processRequest(request);

  for (auto value: parsed_response.values_) {
    std::cout << " === parsed_response return value is " << value << std::flush << std::endl;
  }

  //auto msg = at::empty({pickler.stack().size()}, {at::kByte});

  std::cout << "Test successful" << std::endl;
}
